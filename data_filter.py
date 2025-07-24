# %%
import pandas as pd
from pathlib import Path


json_path = Path.home() / "data/ckp/hf_models/Phi-4-multimodal-instruct/vllm_ls_train_960hr_results.json"
df = pd.read_json(json_path)
df = df[df["WER"] > 0]
# %%
df[["id", "WER"]].to_json(json_path.with_name("ls_train_bad_ids.jsonl"), orient="records", lines=True)
# Example usage:
# filtered = load_and_filter_json(test_json)
# print(filtered)

# %%
