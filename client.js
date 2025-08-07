const BASE_URL = "https://square-macaque-thankfully.ngrok-free.app";

const recordBtn = document.getElementById("record-btn");
const statusDiv = document.getElementById("status");
const inputTextDiv = document.getElementById("input-text");
const responseTextDiv = document.getElementById("response-text");


let mediaRecorder,
    isRecording = false, // 録音中かどうか
    audioChunks = []; // 録音音声データ

// 録音ボタン
recordBtn.addEventListener("click", () => {
    if (!isRecording) {
        startRecording();
        recordBtn.innerText = "録音停止";
        statusDiv.innerText = "録音中";
    } else {
        stopRecording();
        recordBtn.innerText = "録音開始";
        statusDiv.innerText = "録音停止中";
    }
    isRecording = !isRecording;
});


async function startRecording() {
    // マイクの使用許可
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    audioChunks = [];

    // 録音データ取得
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);

    mediaRecorder.onstop = async () => {
        // webmからblobに変換
        const blob = new Blob(audioChunks, { type: "audio/webm" });
        // blobからArrayBufferに変換
        const arrayBuffer = await blob.arrayBuffer();
        // ArrayBufferからwav形式に変換
        const wavBlob = await convertToWav(arrayBuffer);

        recordBtn.disabled = true;
        statusDiv.innerText = "送信中";

        const formData = new FormData();
        formData.append("file", wavBlob, "recorded.wav");

        // 送信
        const response = await fetch(`${BASE_URL}/llm`, {
            method: "POST",
            body: formData,
        });

        if (!response.body) {
            statusDiv.innerText = "エラー: 応答なし";
            recordBtn.disabled = false;
            return;
        }

        // 受信
        statusDiv.innerText = "受信中";
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "",
            audioQueue = [];

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // ストリーム形式でJSONデータが来る
            while (buffer.includes("\n")) {
                const [line, ...rest] = buffer.split("\n");
                buffer = rest.join("\n");
                if (!line.trim()) continue;

                try {
                    const data = JSON.parse(line);
                    if (data.input || data.text) {
                        const inputLine = data.input ? "あなた: " + data.input + "\n" : "";
                        const responseLine = data.text ? "AI: " + data.text + "\n\n" : "";
                        responseTextDiv.innerText =
                            inputLine + responseLine + responseTextDiv.innerText;
                    }

                    // 音声データの受信
                    if (data.voice) {
                        const binary = atob(data.voice);
                        const bytes = new Uint8Array(binary.length);
                        for (let i = 0; i < binary.length; i++)
                            bytes[i] = binary.charCodeAt(i);
                        // 順次再生できるようにキューに追加
                        audioQueue.push(new Blob([bytes], { type: "audio/wav" }));
                    }
                } catch (e) {
                    console.error("JSON parse error", e);
                    continue;
                }
            }
        }

        // 全部の音声データを再生
        for (const blob of audioQueue) await playAudio(blob);
        statusDiv.innerText = "完了";
        recordBtn.disabled = false;
    };

    // 録音開始
    mediaRecorder.start();
}

function stopRecording() {
    mediaRecorder.stop();
}

async function playAudio(blob) {
    return new Promise((resolve) => {
        const audio = new Audio(URL.createObjectURL(blob));
        audio.onended = resolve;
        audio.onerror = resolve;
        audio.play();
    });
}

// 参照　https://qiita.com/tomoyamachi/items/8ff30c3901faa97efb46
async function convertToWav(arrayBuffer) {
    const audioContext = new AudioContext();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const samples = audioBuffer.getChannelData(0);
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, str) =>
        [...str].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));

    writeString(0, "RIFF");
    // file length
    view.setUint32(4, 36 + samples.length * 2, true);
    // set file size at the end
    writeString(8, "WAVE");
    // FMT sub-chunk
    writeString(12, "fmt ");
    // chunk size
    view.setUint32(16, 16, true);
    // format code
    view.setUint16(20, 1, true);
    // channels : (current use monoral : 1)
    view.setUint16(22, 1, true);
    // sampling rate
    view.setUint32(24, audioBuffer.sampleRate, true);
    // data rate
    view.setUint32(28, audioBuffer.sampleRate * 2, true);
    // data block size, (channnel=1)*2=2)
    view.setUint16(32, 2, true);
    // bits per sample (bitDepth)
    view.setUint16(34, 16, true);
    // data sub-chunk
    writeString(36, "data");
    // DUMMY data chunk length (set real value on export)
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let s of samples) {
        const val = Math.max(-1, Math.min(1, s));
        view.setInt16(offset, val < 0 ? val * 0x8000 : val * 0x7fff, true);
        offset += 2;
    }

    return new Blob([view], { type: "audio/wav" });
}

async function submitParams() {
    const speed = parseFloat(document.getElementById("speed").value);
    const volume = parseFloat(document.getElementById("volume").value);

    const response = await fetch(`${BASE_URL}/jtalk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed, volume }),
    });

    const result = await response.json();
    document.getElementById(
        "param"
    ).textContent = `パラメータを変更しました: スピード=${result.speed}, ボリューム=${result.volume}`;
}

async function resetMessages() {
    const res = await fetch(`${BASE_URL}/init`, { method: "POST" });
    const result = await res.json();
    document.getElementById("initLog").textContent = "会話履歴を初期化しました。";
    inputTextDiv.innerText = "";
    responseTextDiv.innerText = "";
}
