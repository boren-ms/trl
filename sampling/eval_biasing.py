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
host = "10.139.201.51"
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
data_path = data_dir / "test-clean-h100.biasing_100.jsonl"
REMOTE_DIR = "az://orngscuscresco/data/boren/data"
ds = ls_bias_dataset(str(data_path), bias_key="distractors", tag=True, num_egs=10, data_dir=REMOTE_DIR)
prompts = []
audios = []
ref_texts = []
fmt = "<|user|>{}<|end|><|assistant|>"
for i, sample in enumerate(ds.select(range(10))):
    print(f"Sample {i}:")  # noqa
    prompt = fmt.format(sample["prompt"][0]["content"])
    print("Prompt", prompt)  # noqa
    print("Text:", sample["text"])  # noqa
    print("Audio Path:", sample["audio_path"])
    prompts.append(prompt)
    audios.append(sample["audio_path"])
    ref_texts.append(sample["text"])
# %%
responses = client.generate(prompts, audios=audios, n=1, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, max_tokens=5120, stop_token_ids=stop_tokens_ids)
for res in responses["texts"]:
    print("Responses:", res)  # noqa

results = []
for res, ref, audio in zip(responses["texts"], ref_texts, audios):
    id = Path(audio).stem
    print("Audio:", audio)  # noqa
    print("Response:", res)  # noqa
    print("Reference:", ref)  # noqa
    results.append({"hyp": res, "ref": ref, "id": id})
# %%|
from trl.scripts.audio_rewards import compute_biasing_metrics
from trl.scripts.audio_metrics import compute_biasing_metrics as new_compute_biasing_metrics

# groups = [[result] for result in results]  # Wrap each result in a list to form groups
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
import json

def load_ref_jsonl(jsonl_path):
    """Load a JSONL file into a dictionary."""
    refs = {}
    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            refs[data["id"]] = {
                "text": data["text"],
                "biasing_words": set(data["ground_truth"]),
            }
    return refs


refs = load_ref_jsonl(data_path)

# %%
import re

def extract_keywords(text):
    """Extract keywords from the text based on biasing words."""
    tagged_words = re.findall(r"\*.*?\*", text)
    keywords  = [wd.strip("*") for wd in tagged_words]
    text = re.sub(r"\*(.*?)\*", r"\1", text)  # Remove tagged words from the text
    return {
        "biasing_words": keywords,
        "text": text,
    }

text = results[-2]["ref"]
print("Original Text:", text)  # noqa
print("Extracted Keywords:", extract_keywords(text))  # noqa

# %%

new_refs = {result["id"]: extract_keywords(result["ref"]) for result in results}
#%%
hyps = {result["id"]: result["hyp"] for result in results}
# %%
wer, uwer, bwer = calc_wers(refs, hyps)
new_wer, new_uwer, new_bwer = calc_wers(new_refs, hyps)

# %%
print(f"WER: {wer.get_result_string()}")
print(f"New WER: {new_wer.get_result_string()}")
print(f"USED WER: {bias_metrics['WER']}")
print(f"U-WER: {uwer.get_result_string()}")
print(f"New U-WER: {new_uwer.get_result_string()}")
print(f"USED U-WER: {bias_metrics['UWER']}")
print(f"B-WER: {bwer.get_result_string()}")
print(f"New B-WER: {new_bwer.get_result_string()}")
print(f"USED B-WER: {bias_metrics['BWER']}")
# %%
# %%
from whisper_normalizer.english import EnglishTextNormalizer
norm = EnglishTextNormalizer()

text = "This is a *test* sentence with *biasing* words."
normalized_text = norm(text)
print("Original Text:", text)  # noqa
print("Normalized Text:", normalized_text)  # noqa

# %%
