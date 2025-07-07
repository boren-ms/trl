# %%
from trl.scripts.report_biasing import calc_wers
import json


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
hyp_file = "/home/boren/data/ckp/hf_models/phi4_mm_bias/ls_biasing/generate_biasing_100_clean.txt"
ref_file = "/home/boren/data/librispeech_biasing/ref/test-clean.biasing_100.jsonl"
hyps = load_hyp_json(hyp_file)
refs = load_ref_jsonl(ref_file)

wer, u_wer, b_wer = calc_wers(refs, hyps)
print("WER:", wer.get_result_string())  # noqa
print("U-WER:", u_wer.get_result_string())  # noqa
print("B-WER:", b_wer.get_result_string())  # noqa

# %%
