# %%
import base64
import json
import requests


def encode_base64_content_from_local_file(file_path: str) -> str:
    """Encode the content of a local file to base64 format."""
    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode("utf-8")
    return result


def call_vllm_chat_completion(audio_str) -> str:

    api_url = "http://localhost:26500/v1/chat/completions"
    headers = {"User-Agent": "Benchmark Client"}
    text = f"Transcribe the spoken language in this audio file into a written document"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{audio_str}"},
                },
            ],
        }
    ]
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

    def get_streaming_response(response: requests.Response):
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"data: "
        ):
            if chunk:
                try:
                    data = json.loads(chunk.decode("utf-8"))
                    choice = data["choices"][0]
                    if "delta" in choice:
                        output = choice["delta"]["content"]
                    else:
                        output = choice["message"]["content"]
                except:
                    continue
                yield output

    output = []
    response = requests.post(
        api_url, headers=headers, json=pload, stream=True, timeout=60
    )
    for h in get_streaming_response(response):
        output.append(h)

    return output


def call_vllm_chat_completion_phi4_mm(
    text: str,
    audio_str: str,
) -> str:

    api_url = "http://localhost:26500/v1/chat/completions"
    headers = {"User-Agent": "Benchmark Client"}
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{audio_str}"},
                },
            ],
        }
    ]
    pload = {
        "messages": messages,
        "n": 1,
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": -1,
        "max_tokens": 200,
        "ignore_eos": False,
        "stream": True,
    }

    def get_streaming_response(response: requests.Response):
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"data: "
        ):
            if chunk:
                try:
                    data = json.loads(chunk.decode("utf-8"))
                    choice = data["choices"][0]
                    if "delta" in choice:
                        output = choice["delta"]["content"]
                    else:
                        output = choice["message"]["content"]
                except:
                    continue
                yield output

    output = []
    response = requests.post(
        api_url, headers=headers, json=pload, stream=True, timeout=60
    )
    for h in get_streaming_response(response):
        output.append(h)

    return "".join(output)


def call_vllm(audio_path: str, prompt: str = None) -> str:
    """Call the vLLM API to transcribe audio."""
    if not audio_path:
        raise ValueError("Audio path must be provided")
    audio_str = encode_base64_content_from_local_file(audio_path)
    if not prompt:
        prompt = (
            "Transcribe the spoken language in this audio file into a written document"
        )

    return call_vllm_chat_completion_phi4_mm(prompt, audio_str)


# %%
# audio_path = "/home/azureuser/cloudfiles/code/Users/vadimma/src/open_asr_leaderboard/wrong_locale2.wav"
audio_path = "/home/boren/data/LibriSpeech/test-clean/2094/142345/2094-142345-0034.flac"
words = [ "cutlery", "utensils", "silverware", "TABLE",  "CLOTHS", "Napkins", "Linen", "dining",  ]
words_text = ", ".join([ f"*{w}*" for w in  words])
text = "Transcribe the audio clip into text."
text = f"{text} Please pay attention to following words: {words_text}."

res = call_vllm(audio_path, prompt=text)
print("Request:")
print(text)
print("Response:")
print(res)
# %%
