import threading

import requests
import time
import pyaudio
import pygame
import base64
import numpy as np
import sounddevice # alsa関連のwarnログが出るのを回避
import io
import wave
import subprocess
import tempfile
import json

import torch
from scipy.io.wavfile import write
from collections import deque
import os

import random
from datetime import datetime
from zoneinfo import ZoneInfo

GPU_SERVER_URL = 'https://square-macaque-thankfully.ngrok-free.app/llm'

FLAG_EMOSUB = 0

SAMPLE_RATE = 16000
#SAMPLE_RATE = 48000
CHANNEL = 1  # チャンネル1(モノラル), ReSpeaker 4 Mic Array = max 6だけど音声検知をリアルタイムで処理するには多いので1で良い
CHUNK_SIZE = 2048 # 1回のデータ取得量
BUFFER_SIZE = 10 # どれくらい貯めるか
SILENCE_DURATION = 1.0  # 発話終了と判定する無音時間

audio = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16

lock = threading.Lock()
event = threading.Event()

buffer = deque(maxlen=BUFFER_SIZE)
recorded_data = []
is_speaking = False
silence_start_time = None

home_dir = os.environ["HOME"]


def load_vad_model():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    return model, get_speech_timestamps


def play_wave(directory, wav_file):
    file_path = os.path.join(directory, wav_file)
    sound = pygame.mixer.Sound(file_path)
    sound.play()
    # 再生が終わるまで待機
    time.sleep(sound.get_length())
    # 再生の終了
    sound.stop()

def save_wave(file, data):
    with wave.open(file, 'wb') as wf:
        wf.setnchannels(CHANNEL)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(data))

class TalkCtrl():
    def __init__(self):
        
        self.shutdown_flag = threading.Event()
        
        pygame.init()
        
        
        self.output_dir = "/home/user/turtlebot3_ws/src/emotion_ros/output_audio/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.filler_dir = "/home/user/turtlebot3_ws/src/emotion_ros/Filler_JP/"
        self.filler_files = os.listdir(self.filler_dir)
        
        self.voice_dir = "/home/user/turtlebot3_ws/src/emotion_ros/Voice_JP/"
        self.hello_wave = "こんにちは_normal.wav"
        

        self.llm_time_flag = 0
        self.filler_flag = 0
        
      
    def play_filler(self):
        random_file = random.choice(self.filler_files)
        if self.filler_flag == 0:
            self.filler_text += random_file
            self.filler_flag += 1
        elif self.filler_flag >= 1:
            self.filler_text += random_file
            self.filler_flag += 1
        play_wave(self.filler_dir, random_file)
    
    def play_jtalk(self, audio_data):
        start_time = time.time()
        pygame.mixer.quit()  # ミキサーを初期化し直す

        # バイト列をWaveオブジェクトとして読み込む
        with io.BytesIO(audio_data) as audio_io:
            with wave.open(audio_io, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                frames = wf.getnframes()
                audio_frames = wf.readframes(wf.getnframes())
                self.llm_len += int(frames / frame_rate)

        # numpyでデータを変換
        if sample_width == 2:
            dtype = np.int16
        else:
            raise ValueError("Unsupported sample width: %d" % sample_width)
        audio_array = np.frombuffer(audio_frames, dtype=dtype)

        # pygame.mixerを適切なフォーマットで初期化
        pygame.mixer.init(frequency=frame_rate, size=-8 * sample_width, channels=n_channels)

        # バイト列に戻してpygameで再生
        tmp = pygame.mixer.Sound(audio_array.tobytes())
        
        tmp.play()

        # 再生が終わるまで待機
        while pygame.mixer.get_busy():
            pygame.time.Clock().tick(10)
    
    def send_audio(self, file_name, output_path):
        def play_filler_loop():
            while not response_event.is_set():
                self.play_filler()
                time.sleep(2.0)

        # レスポンス受信完了を監視するためのイベントフラグ
        response_event = threading.Event()

        # 相槌スレッドを起動
        filler_thread = threading.Thread(target=play_filler_loop, daemon=True)
        filler_thread.start()

        file = {'file': (f"{file_name}", open(f"{output_path}", 'rb'), 'audio/wav')}

        try:
            response = requests.post(GPU_SERVER_URL, stream=True, files=file)
            file['file'][1].close()
            

            if response.status_code == 200:
                buffer = ""
                for chunk in response.iter_content(chunk_size=None):
                    buffer += chunk.decode('utf-8')
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        try:
                            data = json.loads(line)
                            if "voice" in data:
                                audio_data = base64.b64decode(data["voice"])
                                response_event.set()  # レスポンス受信の通知
                                filler_thread.join()
                                
                                if self.llm_time_flag == 0:
                                    self.llm_time_flag = 1
                                
                                self.play_jtalk(audio_data)
                            elif "text" in data:
                                self.get_logger().info(f"認識した音声: {data['input']}")
                                self.get_logger().info(f"返事: {data['text']}")
                        except json.JSONDecodeError:
                            # jsonが途中
                            self.get_logger().info(f"json streaming")
                            continue
            else:
                self.get_logger().info(f"Error: {response.status_code}")
        except Exception as e:
            self.get_logger().info(f"Request failed: {e}")
        finally:
            response_event.set()
            print('finally')
            
    # VADの判定
    def voice_ad(self):
        global is_speaking, silence_start_time
        
        model, get_speech_timestamps = load_vad_model()
        while not self.shutdown_flag.is_set():
            time.sleep(0.1)  # チェック間隔
            with lock:
                if len(buffer) > 0:
                    # 音声データを処理
                    audio_tensor = torch.from_numpy(np.concatenate(buffer)).float() / 32768.0  # 標準化

                    # 発話のタイムスタンプ
                    speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=SAMPLE_RATE, threshold=0.5)

                    if speech_timestamps:
                        # 発話開始
                        if not is_speaking:
                            self.get_logger().info("Thread B: 発話検知、録音開始")
                            start_time = time.time()
                            for buf_data in buffer:
                                recorded_data.append(buf_data.tobytes())
                        is_speaking = True # 発話中
                        silence_start_time = None  # 無音時間リセット
                    else:
                        # 無音が一定時間継続した場合に発話終了と判定
                        if is_speaking:
                            if silence_start_time is None:
                                silence_start_time = time.time()
                            elif (time.time() - silence_start_time >= SILENCE_DURATION) or (time.time() - start_time >= 29):
                                self.get_logger().info("Thread B: 発話終了")
                                is_speaking = False
                                buffer.clear()
                                event.set()  # 発話終了のシグナルを送信
                        else:
                            silence_start_time = None  # 無音時間リセット

    # メインのループ
    def audio_record(self):
        global recorded_data, FLAG_EMOSUB
        file_count = None
        
        stream = audio.open(format=FORMAT, channels=CHANNEL, rate=SAMPLE_RATE, 
                            input=True, input_device_index=3,frames_per_buffer=CHUNK_SIZE)
        self.get_logger().info("Thread A: マイク起動")
        try:
            play_wave(self.voice_dir, self.hello_wave)
            while not self.shutdown_flag.is_set():
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                with lock:
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    if is_speaking:
                        recorded_data.append(data)
                    buffer.append(audio_array)

                # 発話終了のシグナルを受信
                if event.is_set():
                    with lock:
                        file_count = datetime.now(ZoneInfo("Asia/Tokyo")).strftime('%Y%m%d_%H%M%S')
                        file_name = f"speech_{file_count}.wav"
                        file_path = self.output_dir + file_name
                        
                        save_wave(file_path, recorded_data)
                        self.get_logger().info(f"Thread A: 保存{file_name}")
                        
                        # 録音した音声をwavファイルで送信
                        self.send_audio(file_name, file_path)
                        recorded_data.clear()
                    event.clear()
                    buffer.clear()
        finally:
            stream.stop_stream()
            stream.close()
    
    def start(self):        
        thread_a = threading.Thread(target=self.audio_record, daemon=True)
        thread_b = threading.Thread(target=self.voice_ad, daemon=True)
        
        thread_a.start()
        thread_b.start()

def main(args=None):
    try:
      talker = TalkCtrl()
      talker.start()
      while True:
          time.sleep(1)
    except KeyboardInterrupt:
      pass
    finally:
        talker.shutdown_flag.set()
        pygame.quit()
        audio.terminate()
        print("終了")

if __name__ == '__main__':
    main()