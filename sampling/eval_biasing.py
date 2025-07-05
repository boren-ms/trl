# %%
from transformers import AutoProcessor
from trl.extras.vllm_client import VLLMClient
from trl.scripts.audio_dataset import ls_bias_dataset
from pathlib import Path

home_dir = Path.home()
# %%
processor = AutoProcessor.from_pretrained(
    home_dir / "data/ckp/hf_models/phi4_mm_bias_merged",
    trust_remote_code=True,
)
stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt").input_ids.flatten().tolist()


# %%
host = "10.133.182.186"
client = VLLMClient(host=host)
# client.init_communicator() # no need for evaluation only
#%%
prompts = ["<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"]
audios = ["/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
responses = client.generate(prompts, audios=audios, n=4, repetition_penalty=1.0, temperature=0.9, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=100, stop_token_ids=stop_tokens_ids)
for res in responses["texts"]:
    print("Responses:", res)  # noqa
# %%
data_dir = home_dir / "data/librispeech_biasing/ref"
data_paths = [
    data_dir / "test-clean.biasing_100.jsonl",
    # data_dir / "test-clean.biasing_500.jsonl",
    # data_dir / "test-clean.biasing_1000.jsonl",
]
import blobfile as bf
REMOTE_DIR="az://orngscuscresco/data/boren/data"
ds = ls_bias_dataset(data_paths, bias_key="ground_truth", tag=True, num_egs=10, data_dir=REMOTE_DIR)
prompts = []
audios = []
fmt = "<|user|>{}<|end|><|assistant|>"
for i, sample in enumerate (ds):
    print(f"Sample {i}:")  # noqa
    print("Prompt", sample["prompt"][0]["content"])  # noqa
    print("Text:", sample["text"])  # noqa
    print("Audio Path:", sample["audio_path"])
    content = sample["prompt"][0]["content"]
    prompts.append(fmt.format(content))
    audios.append(sample["audio_path"])
    # break
# %%
responses = client.generate(prompts, audios=audios, n=1, repetition_penalty=1.0, temperature=0.9, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=100, stop_token_ids=stop_tokens_ids)
for res in responses["texts"]:
    print("Responses:", res)  # noqa
# %%
