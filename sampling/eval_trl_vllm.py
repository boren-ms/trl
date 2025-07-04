# %%
from transformers import AutoProcessor
import pandas as pd
from trl.extras.vllm_client import VLLMClient


def load_wav_path(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", names=["id", "wav_paths", "msgs"])
    df["wav_path"] = df["wav_paths"].apply(lambda x: eval(x)[0])
    return df["wav_path"].tolist()


# %%
model_path = "/root/data/ckp/hf_models/phi4_mm_bias_merged"
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)
stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt").input_ids.flatten().tolist()


# %%
wav_paths = ["/root/data/LibriSpeech/test-clean/2094/142345/2094-142345-0034.flac", "/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0026.flac"]
words = ["cutlery", "utensils", "silverware", "TABLE", "CLOTHS", "Napkins", "Linen", "dining"]
words_text = ", ".join([f"*{w}*" for w in words])
text = "Transcribe the audio clip into text."
# text = f"{text} Please pay attention to following words: {words_text}."
prompts = []
audios = []
wav_paths = load_wav_path("/root/data/LibriSpeech/debug.tsv")
N=20
for wav_path in wav_paths[:N]:
    prompts.append(f"<|user|><|audio_1|>{text}<|end|><|assistant|>")
    audios.append(wav_path)
#%%
client = VLLMClient()
# client.init_communicator()
prompts = ["<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"]
audios = ["/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
# prompts = ["Hello, AI!", "Tell me a joke"]
# audios = None
# %%
responses = client.generate(prompts, audios=audios, n=4, repetition_penalty=1.0, temperature=0.9, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=100, stop_token_ids=stop_tokens_ids)

for res in responses["texts"]:
    print("Responses:", res)  # noqa
# Update model weights

# %%
