#%%
import blobfile as bf
from pathlib import Path
import pandas as pd
import json
#%%
remote_home = "az://orngscuscresco/data/boren"
rel_model_dir = "data/ckp/hf_models/phi4_mm_bias_merged"
local_model_dir= Path.home() / rel_model_dir
remote_model_dir= f"{remote_home}/{rel_model_dir}"
#%%
results = []
for file in bf.glob(f"{remote_model_dir}/*metrics.json"):
    data = json.loads(bf.BlobFile(file, "r").read())
    for k, v in data.items():
        if not k.startswith("metric"):
            continue
        metric, dataset = k.split("/")
        results.append({
            "metric": metric,
            "dataset": dataset,
            "value": v
        })


df = pd.DataFrame(results)

hf_df = df[df["metric"] == "metric_hf"].drop(columns=["metric"])
vllm_df = df[df["metric"] == "metric_vllm"].drop(columns=["metric"])
diff_df = hf_df.merge(vllm_df, on=["dataset"], suffixes=("_hf", "_vllm"))
diff_df["WERR"] = (diff_df["value_hf"] - diff_df["value_vllm"]) / diff_df["value_hf"]
# %%
reg_df = diff_df[diff_df["WERR"] <0]
reg_df = reg_df.sort_values(by="WERR", ascending=True)
#%%
# print(reg_df)
#             dataset  value_hf  value_vllm      WERR
# 1   clean_1000_UWER  1.094207    1.523473 -0.392308
# 0    clean_1000_WER  1.338890    1.791473 -0.338028
# 2   clean_1000_BWER  3.450781    4.104613 -0.189474
# 17   other_500_BWER  6.060012    6.354187 -0.048544
# 14   other_100_BWER  4.052524    4.157678 -0.025948
# 8    clean_500_BWER  2.579005    2.633491 -0.021127
# 19    other_no_UWER  2.703777    2.760280 -0.020898
# 5    clean_100_BWER  1.598256    1.616418 -0.011364
# 10    clean_no_UWER  1.167856    1.176273 -0.007207
# 9      clean_no_WER  1.851817    1.853703 -0.001018
# %%
# check clean_1000_WER results


#%%
dfs = {}
for file in bf.scanglob(f"{remote_model_dir}/*_clean_1000_results.json"):
    name = file.name
    print(f"Processing {file}")
    data = json.loads(bf.BlobFile(file.path, "r").read())
    df = pd.DataFrame(data)
    dfs[name] = df
#%%
from jiwer import wer
from whisper_normalizer.english import EnglishTextNormalizer

# Instead of plain pd.concat, join with suffixes from keys
def compute_wer(hyp, ref):
    norm = EnglishTextNormalizer()
    hyp = norm(hyp)
    ref = norm(ref)
    return wer(hyp, ref)

hf_res = dfs["hf_clean_1000_results.json"]
vllm_res = dfs["vllm_clean_1000_results.json"]
#%%
hf_res["wer"] = hf_res.apply(lambda x: compute_wer(x["hyp"], x["ref"]), axis=1)
vllm_res["wer"] = vllm_res.apply(lambda x: compute_wer(x["hyp"], x["ref"]), axis=1)
#%%
res_df = hf_res.merge(vllm_res, on="id", suffixes=("_hf", "_vllm"))
# %%
res_df["werr"] = 1 - res_df["wer_vllm"] / (res_df["wer_hf"].replace(0, 1e-6))
res_diff = res_df[res_df["werr"] < 0].sort_values(by="werr", ascending=True)
res_diff = res_diff[["id", "hyp_hf", "hyp_vllm", "ref_hf", "wer_hf", "wer_vllm", "werr"]]
res_diff = res_diff.rename(columns={"ref_hf": "ref"})
# %%
# Write res_diff to JSONL (one JSON object per line)
diff_jsonl_file= f"{remote_model_dir}/clean_1000_hf_vllm_diff.jsonl"
with bf.BlobFile(diff_jsonl_file, "w") as f:
    res_diff.to_json(f, orient="records", lines=True)
print(f"Wrote {len(res_diff)} differences to {diff_jsonl_file}")
# %%

