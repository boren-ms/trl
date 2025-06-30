import os
import json
import asyncio
import aiohttp
import librosa
import numpy as np
import base64
from jiwer import wer
import requests
from tqdm import tqdm

# CONFIG
AUDIO_DIR = "/home/azureuser/cloudfiles/code/Users/vadimma/dataset-ruchao/team4_LX/wav"
JSONL_PATH = "/home/azureuser/cloudfiles/code/Users/vadimma/dataset-ruchao/team4_LX/test.jsonl"
VLLM_ENDPOINT = "http://localhost:8000/transcribe"  # change to your VLLM endpoint
CONCURRENT_REQUESTS = 10  # adjust as needed


async def encode_base64_content_from_local_file(file_path: str) -> str:
    """Encode the content of a local file to base64 format."""
    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode('utf-8')
    return result

async def call_vllm_chat_completion(session, audio_id, audio_path):
    # Encode the audio file in base64
    audio_str = await encode_base64_content_from_local_file(audio_path)
    api_url = "http://localhost:26500/v1/chat/completions"
    headers = {"User-Agent": "Benchmark Client"}
    text = f"Transcribe the spoken language in this audio file into a written document"
    messages = [{ "role": "user", "content": [
            {
                "type": "text",
                "text": text
            },
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/wav;base64,{audio_str}"
                },
            },
    ],
    }]
    pload = {
        "messages": messages,
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 200,
        "ignore_eos": False,
        "stream": True,
    }

    # Send the request and process the streaming response
    async with session.post(api_url, headers=headers, json=pload) as resp:
        resp.raise_for_status()
        output = []
        async for line in resp.content:
            if line.startswith(b"data: "):
                chunk = line[len(b"data: "):].strip()
                if chunk:
                    try:
                        data = json.loads(chunk.decode("utf-8"))
                        choice = data['choices'][0]
                        if 'delta' in choice:
                            output.append(choice['delta']['content'])
                        else:
                            output.append(choice['message']['content'])
                    except Exception:
                        continue

    return audio_id, ''.join(output)

# Load ground truth transcriptions
def load_transcriptions(jsonl_path):
    transcripts = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            transcripts[os.path.basename(obj['WavPath'])] = obj['Transcription']
    return transcripts

# Load and encode audio as base64
def load_audio_base64(path):
    y, sr = librosa.load(path, sr=16000)
    audio_bytes = (y * 32767).astype(np.int16).tobytes()
    b64 = base64.b64encode(audio_bytes).decode('utf-8')
    return b64

# Async request to VLLM
async def transcribe(session, audio_id, audio_path):
    audio_id, transcription = await call_vllm_chat_completion(session, audio_id, audio_path)
    return audio_id, transcription


# Main processing logic
async def run_inference(audio_dir, transcripts):
    results = {}
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        async def sem_task(audio_id, audio_path):
            async with semaphore:
                try:
                    return await transcribe(session, audio_id, audio_path)
                except Exception as e:
                    print(f"[Error] {audio_id}: {e}")
                    return audio_id, ""

        tasks = []
        for audio_id in transcripts:
            audio_file = os.path.join(audio_dir, f"{audio_id}")
            if not os.path.exists(audio_file):
                print(f"[Warning] Missing audio file: {audio_file}")
                continue
            tasks.append(sem_task(audio_id, audio_file))

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            audio_id, hyp = await coro
            results[audio_id] = hyp

    return results

# Compute WER
def compute_wer(references, hypotheses):
    ref_list = [references[k] for k in references if k in hypotheses]
    hyp_list = [hypotheses[k] for k in references if k in hypotheses]
    return wer(ref_list, hyp_list)

# Run everything
def main():
    print("Loading transcriptions...")
    references = load_transcriptions(JSONL_PATH)

    print("Running inference...")
    hypotheses = asyncio.run(run_inference(AUDIO_DIR, references))

    print("Computing WER...")
    error_rate = compute_wer(references, hypotheses)
    print(f"Word Error Rate (WER): {error_rate:.2%}")

if __name__ == "__main__":
    main()
