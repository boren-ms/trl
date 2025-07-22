# %%
import requests
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
# %%
prompts = [val["prompt"] for val in inputs.values()]
audios = [val["audio"] for val in inputs.values()]
ref_texts = [val["ref"] for val in inputs.values()]

# %%
from more_itertools import chunked
from tqdm import tqdm
n = 2
chunk_size = 2400
chunks = list(chunked(zip(audios, prompts), chunk_size))
responses = []
for chunk in tqdm(chunks, "vllm inference..."):
    chunk_audios, chunk_prompts = zip(*chunk)
    chunk_responses = client.generate(
        chunk_prompts,
        audios=chunk_audios,
        n=n,
        temperature=0.000001,
        max_tokens=512,
        stop_token_ids=stop_tokens_ids,
        # generation_kwargs={"do_sample": False},
    )
    responses.extend(chunk_responses["texts"])

#%%
n = 2

results = []
for res, ref, audio in zip(responses[::n], ref_texts, audios):
    id = Path(audio).stem
    results.append({"hyp": res, "ref": ref, "id": id})

# %%
import json

result_jsonl = model_path / "vllm_results.jsonl"
with open(result_jsonl, "w") as f:
    for result in results:
        f.write(json.dumps(result, separators=(",", ":")) + "\n")
#%%
from trl.scripts.audio_metrics import compute_wers

wer, u_wer, b_wer = compute_wers(results)
print("WER:", wer.get_result_string())  # noqa
print("U-WER:", u_wer.get_result_string())  # noqa
print("B-WER:", b_wer.get_result_string())  # noqa

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
#%%


keys = list(refs.keys())
for key in keys:
    wer, u_wer, b_wer = new_calc_wers({key: refs[key]}, {key: hyps[key]})
    new_wer, new_u_wer, new_b_wer = new_calc_wers({key: refs[key]}, {key: new_hyps[key]})
    if wer.get_wer() != new_wer.get_wer():
        print(f"Key: {key}")
        print("Bias Words:", refs[key].get("biasing_words", "N/A"))
        print("Reference            :", refs[key]["text"])
        print("Hypothesis (old hyps):", hyps[key])
        print("Hypothesis (new hyps):", new_hyps[key])
        print("-" * 40)
# %%
