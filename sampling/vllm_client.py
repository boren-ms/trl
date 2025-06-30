# %%
import base64
import requests


def encode_base64_content_from_local_file(file_path):
    """Encode the content of a local file to base64 format."""
    with open(file_path, "rb") as file:
        file_content = file.read()
        result = base64.b64encode(file_content).decode("utf-8")
    return result


def vllm_completion(text, audio_str, **kwargs):

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
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": 200,
        "ignore_eos": False,
    }
    pload.update(kwargs)

    response = requests.post(api_url, headers=headers, json=pload, timeout=60)
    if response.status_code == 200:
        return [ choice["message"]["content"] for choice in response.json()["choices"]]
    else:
        raise ValueError(f"Request failed: {response.status_code}, {response.text}")


def call_vllm(audio_path, prompt=None, **kwargs):
    """Call the vLLM API to transcribe audio."""
    if not audio_path:
        raise ValueError("Audio path must be provided")
    audio_str = encode_base64_content_from_local_file(audio_path)
    if not prompt:
        prompt = (
            "Transcribe the spoken language in this audio file into a written document"
        )

    return vllm_completion(prompt, audio_str, **kwargs)


# %%
audio_path = "/home/boren/data/LibriSpeech/test-clean/2094/142345/2094-142345-0034.flac"
words = [
    "cutlery",
    "utensils",
    "silverware",
    "TABLE",
    "CLOTHS",
    "Napkins",
    "Linen",
    "dining"
]
words_text = ", ".join([f"*{w}*" for w in words])
text = "Transcribe the audio clip into text."
text = f"{text} Please pay attention to following words: {words_text}."

res = call_vllm(audio_path, prompt=text, n=12, temperature=0.95, top_p=0.95, max_tokens=200)
print("Request:")
print(text)
print("Response:")
for i, r in enumerate(res):
    print(f"[{i+1:02d}]:", r)
# %%
