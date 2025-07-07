# %%
from fastapi import requests
from transformers import AutoProcessor
from trl.extras.vllm_client import VLLMClient
from trl.scripts.audio_dataset import ls_bias_dataset
from pathlib import Path

home_dir = Path.home()
# %%
model_path = home_dir / "data/ckp/hf_models/phi4_mm_bias_merged"
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)
stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt").input_ids.flatten().tolist()


# %%
host = "10.139.241.19"
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
REMOTE_DIR = "az://orngscuscresco/data/boren/data"
ds = ls_bias_dataset(str(data_path), bias_key="distractors", tag=True, data_dir=REMOTE_DIR)
prompts = []
audios = []
ref_texts = []
fmt = "<|user|>{}<|end|><|assistant|>"
inputs = {}
for i, sample in enumerate(ds):
    id = Path(sample["audio_path"]).stem
    inputs[id] = {"prompt": fmt.format(sample["prompt"][0]["content"]), "audio": sample["audio_path"], "ref": sample["text"]}
#%%
prompts = [val["prompt"] for val in inputs.values()]
audios = [val["audio"] for val in inputs.values()]
ref_texts = [val["ref"] for val in inputs.values()]

# %%
from more_itertools import chunked
chunk_size = 100

responses = []
for chunk in chunked(zip(audios, prompts), chunk_size):
    chunk_audios, chunk_prompts = zip(*chunk)
    chunk_responses = client.generate(chunk_prompts, audios=chunk_audios, n=1, temperature=0, max_tokens=512, stop_token_ids=stop_tokens_ids)
    responses.extend(chunk_responses["texts"])

results = []
for res, ref, audio in zip(responses, ref_texts, audios):
    id = Path(audio).stem
    results.append({"hyp": res, "ref": ref, "id": id})

# %%
import json

result_jsonl = model_path / "vllm_results.jsonl"
with open(result_jsonl, "w") as f:
    for result in results:
        f.write(json.dumps(result, separators=(",", ":")) + "\n")
# %%|
# from trl.scripts.audio_rewards import compute_biasing_metrics
from trl.scripts.audio_metrics import compute_biasing_metrics, compute_wers

# bias_metrics = compute_biasing_metrics(groups)
# print("Bias Metrics:", bias_metrics["WER"], bias_metrics["UWER"], bias_metrics["BWER"])
# new_bias_metrics = compute_biasing_metrics(results)
# print("New Bias Metrics:", new_bias_metrics["WER"], new_bias_metrics["UWER"], new_bias_metrics["BWER"])

wer, u_wer, b_wer = compute_wers(results)
print("WER:", wer.get_result_string())  # noqa
print("U-WER:", u_wer.get_result_string())  # noqa
print("B-WER:", b_wer.get_result_string())  # noqa

# %%
for i, result in enumerate(results):
    print(f"Result {i}:")  # noqa
    print("ID:", result["id"])  # noqa
    print("Hypothesis:", result["hyp"])  # noqa
    print("Reference:", result["ref"])  # noqa
    bias_metrics = compute_biasing_metrics([[result]])
    print("Bias Metrics:", bias_metrics["WER"], bias_metrics["UWER"], bias_metrics["BWER"])
    new_bias_metrics = compute_biasing_metrics([result])
    print("New Bias Metrics:", new_bias_metrics["WER"], new_bias_metrics["UWER"], new_bias_metrics["BWER"])

# %%


def load_ref_jsonl(ref_file):
    """Load reference and hypothesis files in JSON format."""
    refs = {}
    with open(ref_file, "r", encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            refs[data["id"]] = {
                "text": data["text"],
                "biasing_words": set(data["ground_truth"]),
            }
    return refs


def load_hyp_json(hyp_file):
    hyps = {}
    with open(hyp_file, "r", encoding="utf8") as f:
        data = json.load(f)[:-1]
        for utt in data:
            hyps[utt["audio_ids"][0]] = utt["generated_texts"][0]
    return hyps


# %%
from trl.scripts.report_biasing import calc_wers

hyp_file = "/home/boren/data/ckp/hf_models/phi4_mm_bias/ls_biasing/generate_biasing_100_clean.txt"
ref_file = "/home/boren/data/librispeech_biasing/ref/test-clean.biasing_100.jsonl"
hyps = load_hyp_json(hyp_file)
refs = load_ref_jsonl(ref_file)
# %%
wer, u_wer, b_wer = calc_wers(refs, hyps)
print("WER:", wer.get_result_string())  # noqa
print("U-WER:", u_wer.get_result_string())  # noqa
print("B-WER:", b_wer.get_result_string())  # noqa
# %%
from trl.scripts.audio_metrics import calc_wers as new_calc_wers
from trl.scripts.audio_metrics import extract_keywords


new_refs = {result["id"]: extract_keywords(result["ref"]) for result in results}
new_hyps = {result["id"]: result["hyp"] for result in results}
# Calculate WER, U-WER, and B-WER
# %%
new_wer, new_u_wer, new_b_wer = new_calc_wers(new_refs, new_hyps)
print("WER:", new_wer.get_result_string())  # noqa
print("U-WER:", new_u_wer.get_result_string())  # noqa
print("B-WER:", new_b_wer.get_result_string())  # noqa
# %%
new_wer, new_u_wer, new_b_wer = new_calc_wers(refs, hyps)
print("WER:", new_wer.get_result_string())  # noqa
print("U-WER:", new_u_wer.get_result_string())  # noqa
print("B-WER:", new_b_wer.get_result_string())  # noqa
# %%
new_wer, new_u_wer, new_b_wer = calc_wers(new_refs, hyps)
print("WER:", new_wer.get_result_string())  # noqa
print("U-WER:", new_u_wer.get_result_string())  # noqa
print("B-WER:", new_b_wer.get_result_string())  # noqa
# %%
# Compare keys and values between new_refs and refs
# Compare keys and values between new_hyps and hyps
new_hyp_keys = set(new_hyps.keys())
hyp_keys = set(hyps.keys())

print("Keys in new_hyps but not in hyps:", new_hyp_keys - hyp_keys)
print("Keys in hyps but not in new_hyps:", hyp_keys - new_hyp_keys)

# Check for differences in values for common keys
common_keys = new_hyp_keys & hyp_keys
for key in common_keys:
    if new_hyps[key] != hyps[key]:
        print(f"Difference for key {key}:")
        print("  new_hyps:", new_hyps[key])
        print("  hyps    :", hyps[key])
# %%
# Evaluate and compare each item from hyps and new_hyps
import pandas as pd
import difflib

diff_items = []
#%%
keys = list(refs.keys())
for key in keys:
    wer, u_wer, b_wer = new_calc_wers({key: refs[key]}, {key: hyps[key]})
    new_wer, new_u_wer, new_b_wer = new_calc_wers({key: refs[key]}, {key: new_hyps[key]})
    if wer.get_wer() != new_wer.get_wer():
        d = difflib.HtmlDiff()
        diff = difflib.unified_diff(
            hyps[key].split(), new_hyps[key].split()
        )
        diff_items.append({
            "id": key,
            "ref": refs[key]["text"],
            "old_hyp": hyps[key],
            "new_hyp": new_hyps[key],
        })
        print(f"Key: {key}")
        print("Bias Words:", refs[key].get("biasing_words", "N/A"))
        print("Reference            :", refs[key]["text"])
        print("Hypothesis (old hyps):", hyps[key])
        print("Hypothesis (new hyps):", new_hyps[key])
        print("-"*40)
# %%
df = pd.DataFrame(diff_items)
print(df.to_markdown(index=False))  # Display the DataFrame in markdown format
# %%
        # # print("Bias Word:", new_refs.get(key, {}).get("biasing_words", "N/A"))
        # print("Hypothesis (old hyps):", old)
        # print("Hypothesis (new hyps):", new)
        # print("Diff:")
        # print("\n".join(diff))
        # print("WER:", wer.get_result_string())
        # print("New WER:", new_wer.get_result_string())
        # print("U-WER:", u_wer.get_result_string())
        # print("New U-WER:", new_u_wer.get_result_string())
        # print("B-WER:", b_wer.get_result_string())
        # print("New B-WER:", new_b_wer.get_result_string())
# %%
df = pd.DataFrame(diff_items)
# %%
print(df.to_markdown(index=False))  # Display the DataFrame in markdown format
# %%

# %%
