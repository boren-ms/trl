import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json

from report_biasing import calc_wers


#%%
def load_result(hyp_file):
    hyps = {}
    refs = {}
    with open(hyp_file, "r", encoding="utf8") as f:
        data = json.load(f)
        for utt in data:
            hyps[utt["id"]] = utt["hyp"]
            refs[utt["id"]] = utt["ref"]
    return hyps, refs


def load_ref(ref_file):
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


#%%
name = "clean"
hyp_file = f"/root/data/ckp/hf_models/phi4_mm_bias_merged/hf_{name}_no_results.json"
ref_file = f"/root/data/librispeech_biasing/ref/test-{name}.biasing_100.jsonl"

refs = load_ref(ref_file)
hyps, _ = load_result(hyp_file)
print(f"Loaded {len(refs)} refs from {ref_file}")
print(f"Loaded {len(hyps)} hyps from {hyp_file}")
#%%
miss = set(refs.keys()) - set(hyps.keys())
if miss:
    print("Missing IDs in hypothesis: %s", miss)
wer, uwer, bwer = calc_wers(refs, hyps)
print(f"WER: {wer.get_result_string()}")
print(f"U-WER: {uwer.get_result_string()}")
print(f"B-WER: {bwer.get_result_string()}")

# %%
