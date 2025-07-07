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
host = "10.142.78.249"
client = VLLMClient(host=host)
# %%
prompts = ["<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"]
audios = ["/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
responses = client.generate(prompts, audios=audios, n=4, repetition_penalty=1.0, temperature=0.9, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=100, stop_token_ids=stop_tokens_ids)
for res in responses["texts"]:
    print("Responses:", res)  # noqa
# %%
data_dir = home_dir / "data/librispeech_biasing/ref"
# data_path = data_dir / "test-clean.biasing_100.jsonl"
data_path = data_dir / "test-clean.biasing_100.jsonl"
REMOTE_DIR = "az://orngwus2cresco/data/boren/data"
ds = ls_bias_dataset(str(data_path), bias_key="distractors", tag=True,  data_dir=REMOTE_DIR)
prompts = []
audios = []
ref_texts = []
fmt = "<|user|>{}<|end|><|assistant|>"
for i, sample in enumerate(ds):
    # print(f"Sample {i}:")  # noqa
    prompt = fmt.format(sample["prompt"][0]["content"])
    # print("Prompt", prompt)  # noqa
    # print("Text:", sample["text"])  # noqa
    # print("Audio Path:", sample["audio_path"])
    prompts.append(prompt)
    audios.append(sample["audio_path"])
    ref_texts.append(sample["text"])
# %%
responses = client.generate(prompts, audios=audios, n=1, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=5120, stop_token_ids=stop_tokens_ids)

results = []
for res, ref, audio in zip(responses["texts"], ref_texts, audios):
    id = Path(audio).stem
    results.append({"hyp": res, "ref": ref, "id": id})
# %%|
# from trl.scripts.audio_rewards import compute_biasing_metrics
from trl.scripts.audio_metrics import compute_biasing_metrics, compute_wers

groups = [[result] for result in results]  # Wrap each result in a list to form groups
# bias_metrics = compute_biasing_metrics(groups)
# print("Bias Metrics:", bias_metrics["WER"], bias_metrics["UWER"], bias_metrics["BWER"])
# new_bias_metrics = compute_biasing_metrics(results)
# print("New Bias Metrics:", new_bias_metrics["WER"], new_bias_metrics["UWER"], new_bias_metrics["BWER"])

wer, u_wer, b_wer = compute_wers(results)
print("WER:", wer.get_result_string())  # noqa
print("U-WER:", u_wer.get_result_string())  # noqa
print("B-WER:", b_wer.get_result_string())  # noqa
    
#%%
for i, result in enumerate(results):
    print(f"Result {i}:")  # noqa
    print("ID:", result["id"])  # noqa
    print("Hypothesis:", result["hyp"])  # noqa
    print("Reference:", result["ref"])  # noqa
    bias_metrics = compute_biasing_metrics([[result]])
    print("Bias Metrics:", bias_metrics["WER"], bias_metrics["UWER"], bias_metrics["BWER"])
    new_bias_metrics = new_compute_biasing_metrics([result])
    print("New Bias Metrics:", new_bias_metrics["WER"], new_bias_metrics["UWER"], new_bias_metrics["BWER"])

# %%
