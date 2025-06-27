import base64
import json
import requests

def encode_base64_content_from_local_file(file_path: str) -> str:
    """Encode the content of a local file to base64 format."""
    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode('utf-8')
    return result


def call_vllm_chat_completion(
    audio_str
) -> str:

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
        "top_p": 0.9,
        "max_tokens": 100,
        "ignore_eos": True,
        "stream": True,
        "model": "speech",
    }

    def get_streaming_response(
        response: requests.Response
    ):
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"data: "
        ):
            if chunk:
                try:
                    data = json.loads(chunk.decode("utf-8"))
                    choice = data['choices'][0]
                    if 'delta' in choice:
                        output = choice['delta']['content']
                    else:
                        output = choice['message']['content']
                except:
                    continue
                yield output

    output = []
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    for h in get_streaming_response(response):
        output.append(h)

    return output

def call_vllm_chat_completion_phi4_mm(
    audio_str
) -> str:

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

    def get_streaming_response(
        response: requests.Response
    ):
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"data: "
        ):
            if chunk:
                try:
                    data = json.loads(chunk.decode("utf-8"))
                    choice = data['choices'][0]
                    if 'delta' in choice:
                        output = choice['delta']['content']
                    else:
                        output = choice['message']['content']
                except:
                    continue
                yield output

    output = []
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    for h in get_streaming_response(response):
        output.append(h)

    return output


# audio_path = "/home/azureuser/cloudfiles/code/Users/vadimma/src/open_asr_leaderboard/wrong_locale2.wav"
audio_path = "/home/boren/data/LibriSpeech/test-clean/2094/142345/2094-142345-0034.flac"
audio_str = encode_base64_content_from_local_file(audio_path)
print(call_vllm_chat_completion_phi4_mm(audio_str))
