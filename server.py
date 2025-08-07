from flask import Flask, request, jsonify, json, Response, send_file
from flask_ipfilter import IPFilter, Whitelist
from flask_cors import CORS

import os
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import io

import torch
import torchaudio
from transformers import pipeline
from vllm import LLM, SamplingParams

import numpy as np
import base64
import json

import re
import unicodedata
from alkana import get_kana
import markdown
from bs4 import BeautifulSoup


app = Flask(__name__)
CORS(app)

SAMPLE_RATE = 16000
#SAMPLE_RATE = 48000

llm = None
tokenizer = None
DEFAULT_SYSTEM_PROMPT = ""
messages_log = []
sampling_params = None
TOKEN_LIMIT = 8192  # トークン数制限
MEMORY_LOG_NUM = 30

pipe = None
generate_kwargs = None

cmd = ''
file_count = 0

UPLOAD_FOLDER = '/home/user/pro/python/llm/speech_audio'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# トークン数を計算
def calculate_token_count(messages):
    global tokenizer
    combined_text = "\n".join([msg["content"] for msg in messages])
    tokens = tokenizer.encode(combined_text)
    return len(tokens)

def token_confirm(messages):
    while calculate_token_count(messages) > TOKEN_LIMIT:
        messages.pop(0)


def initialize_models():
    start_time = time.time()
    global llm, tokenizer, DEFAULT_SYSTEM_PROMPT, sampling_params, pipe, generate_kwargs
    # LLMを初期化
    llm = LLM(model="elyza/Llama-3-ELYZA-JP-8B-AWQ", quantization="awq_marlin")
    tokenizer = llm.get_tokenizer()

    with open("./prompt_default.txt", "r", encoding="utf-8") as f:
        DEFAULT_SYSTEM_PROMPT = f.read()
    
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1000)
    
    model_id = "kotoba-tech/kotoba-whisper-v2.0"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
    generate_kwargs = {
        "language": "japanese",
        "task": "transcribe",
        "return_timestamps": True # 30秒以上の音声だとTrue
    }

    
    # 音声認識の初期化
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs=model_kwargs
    )
    end_time = time.time()
    print("init time: {:.2f} seconds".format(end_time - start_time))

def init_jtalk():
    global cmd
    open_jtalk = ['open_jtalk']
    mech = ['-x', '/var/lib/mecab/dic/open-jtalk/naist-jdic']
    htsvoice = ['-m', '/usr/share/hts-voice/mei/mei_normal.htsvoice']
    speed = ['-r', '1.0']
    volume=['-g','1.0']
    cmd = open_jtalk + mech + htsvoice + speed + volume

init_jtalk()
initialize_models()

@app.route('/jtalk', methods=['POST'])
def edit_jtalk_param():
    global cmd
    try:
        data = request.get_json()
        speed_value = float(data.get('speed', 1.0))
        volume_value = float(data.get('volume', 1.0))

        open_jtalk = ['open_jtalk']
        mech = ['-x', '/var/lib/mecab/dic/open-jtalk/naist-jdic']
        htsvoice = ['-m', '/usr/share/hts-voice/mei/mei_normal.htsvoice']
        speed = ['-r', str(speed_value)]
        volume = ['-g', str(volume_value)]
        outwav = ['-ow', '/dev/stdout']
        cmd = open_jtalk + mech + htsvoice + speed + volume + outwav

        return jsonify({
            "message": "パラメータが設定されました",
            "speed": speed_value,
            "volume": volume_value,
            "cmd": cmd
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/init', methods=['GET', 'POST'])
def init_msg_log():
    global messages_log
    if request.method == 'POST':
        messages_log = []
        return jsonify({"message": "会話履歴を初期化しました。"})
    else:
        return jsonify({"message_log": list(messages_log)})



@app.route('/llm', methods=['GET', 'POST'])
def receive_message():
    global messages_log

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.wav'):
        save_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
        file.save(save_path)
    
    input_text = recognize_speech(save_path)

    messages_log.append({"role": "user", "content": input_text})
    
    token_confirm(messages_log)

    recent_history = messages_log[-(MEMORY_LOG_NUM*2):]  # 最新NUM個

    
    # 応答を生成
    speech = generate_speech(recent_history)
    messages_log.append({"role": "assistant", "content": speech})
    
    text = md_to_text(speech)
    lines = parse_lines(text)
    # lines = [L for L in lines if includes_japanese(L)]
    lines = [convert_english_words(L) for L in lines]
    lines = "".join(lines)
    lines = split_sentences(lines)
    print(lines)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(jtalk_bytes, line): i for i, line in enumerate(lines)}
        results = [None] * len(lines)
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            
    print(len(results))

    def generate_response():
        initial_response = json.dumps({"text": speech, "input": input_text}) + "\n"
        yield initial_response

        print(len(results))
        for result in results:
            voice_encoded = base64.b64encode(result).decode('utf-8')
            yield json.dumps({"voice": voice_encoded}) + "\n"

    return Response(generate_response(), content_type='application/json')

def generate_speech(recent_history):
    start_time = time.time()

    messages_batch = [
        [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ] + recent_history
    ]

    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]

    # LLMで応答を生成
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    print("生成時間: {:.2f} seconds".format(end_time - start_time))

    for output in outputs:
        return output.outputs[0].text

def recognize_speech(audio_file):
    start_time = time.time()

    # 一時ファイルから音声をロード
    waveform, sample_rate = torchaudio.load(audio_file)

    # サンプリングレートをリサンプル
    resample_waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)

    # PyTorchテンソルをNumPy配列に変換
    resample_waveform_np = resample_waveform.numpy()

    # 音声認識の実行
    result = pipe({"array": resample_waveform_np[0], "sampling_rate": SAMPLE_RATE}, generate_kwargs=generate_kwargs)
    
    end_time = time.time()

    print(result["text"])
    print("認識時間: {:.2f} seconds".format(end_time - start_time))
    
    return result["text"]

def jtalk_bytes(t):
    global cmd, file_count

    start_time = time.time()

    file_count += 1
    outwav = ['-ow', '/dev/stdout']
    full_cmd = cmd + outwav

    # 音声生成（標準出力に書き出し）
    c = subprocess.Popen(full_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_data, _ = c.communicate(input=t.encode('utf-8'))

    end_time = time.time()
    print("合成時間: {:.2f} seconds".format(end_time - start_time))

    return audio_data



# 参照 https://stackoverflow.com/questions/761824/python-how-to-convert-markdown-formatted-text-to-text
def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()
    
    
def is_japanese(ch):
    name = unicodedata.name(ch, None)
    return name is not None and \
            ("CJK UNIFIED" in name or "HIRAGANA" in name or "KATAKANA" in name)


def includes_japanese(s):
    return any(is_japanese(ch) for ch in s)


def convert_english_words(L):
    ws = re.split(r"([a-zA-Z']+)", L)
    ys = []
    for w in ws:
        if re.fullmatch("[a-zA-Z']+", w):
            w = get_kana(w) or get_kana(w.lower()) or w
        ys.append(w)
    return ''.join(ys)


def parse_lines(text):
    text = re.sub(r'\n\s*(?<=[a-zA-Z])', ' ', text)
    text = re.sub(r'\n\s*(?<=[^a-zA-Z])', '', text)
    text = text.replace('\n\n', '。')

    # remove spaces between zenkaku and hankaku
    text = re.sub(r'((?!\n)\s)+', ' ', text)
    s = list(text)
    for i in range(1, len(text) - 1):
        prev_ch, ch, next_ch = s[i-1], s[i], s[i+1]
        if ch == ' ':
            prev_ch_is_japanese = is_japanese(prev_ch)
            next_ch_is_japanese = is_japanese(next_ch)
            if prev_ch_is_japanese and not next_ch_is_japanese or not prev_ch_is_japanese and next_ch_is_japanese:
                s[i] = ''
    text = ''.join(s)

    lines = re.split(r"([。\n]+)", text)
    r = []
    for L in lines:
        if not L:
            pass
        elif re.fullmatch(r"[。\n]+", L):
            if r and "。" in L:
                r[-1] += "。"
        else:
            r.append(L)
    lines = r

    return lines

def split_sentences(text, max_length=250):
    sentences = re.split(r'(\。|!|！|\?|？)', text)

    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                 for i in range(0, len(sentences), 2)]

    merged_text = []
    current_chunk = ''

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence
        else:
            if current_chunk:
                merged_text.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        merged_text.append(current_chunk)

    return merged_text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)