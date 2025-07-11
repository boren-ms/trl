# %%
import pandas as pd
import json
from pathlib import Path
from jiwer import compute_measures
from whisper_normalizer.english import EnglishTextNormalizer
import blobfile as bf


def compute_errors(ref, hyp):
    """Compute WER and other error metrics between reference and hypothesis."""
    measures = compute_measures(ref, hyp)
    err = measures["substitutions"] + measures["deletions"] + measures["insertions"]
    return {"wer": measures["wer"], "errors": err}


def load_result(json_path):
    with bf.BlobFile(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["id"], keep="first")
    norm = EnglishTextNormalizer()
    df["hyp"] = df["hyp"].apply(norm)
    df["ref"] = df["ref"].apply(norm)
    df[["wer", "errors"]] = pd.DataFrame(df.apply(lambda x: compute_errors(x["ref"], x["hyp"]), axis=1).tolist(), index=df.index)
    return df


# %%
data_dir = Path.home() / "data/ckp/hf_models/phi4_mm_bias_merged/"
# data_dir = "az://orngwus2cresco/data/boren/data/ckp/hf_models/phi4_mm_bias_merged/"
json_name = "clean_1000_results.json"
hf_df = load_result(f"{data_dir}/hf_{json_name}")
vllm_df = load_result(f"{data_dir}/vllm_{json_name}")
# %%

df = hf_df.merge(vllm_df, on="id", suffixes=("_hf", "_vllm"))

df["werr"] = df.apply(lambda x: 1 - x["wer_vllm"] / (x["wer_hf"] + 1e-8), axis=1)
df["err"] = df.apply(lambda x: x["errors_vllm"] - x["errors_hf"], axis=1)


diff_df = df[df["err"] > 0]

diff_df = diff_df.sort_values(by="err", ascending=False)
# %%
for index, row in diff_df.iterrows():
    print(f"ID: {row['id']}, WERR: {row['werr']:.2%}, WER HF: {row['wer_hf']:.2%}, WER VLLM: {row['wer_vllm']:.2%}")
    print(f"     REF: {row['ref_hf']}")
    print(f"HF   HYP: {row['hyp_hf']}")
    print(f"VLLM HYP: {row['hyp_vllm']}")
    print("-" * 80)
# %%
