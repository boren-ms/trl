#%%
import pandas as pd
import json
from pathlib import Path
from jiwer import wer
from whisper_normalizer.english import EnglishTextNormalizer

def load_result(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=['id'], keep='first')
    norm = EnglishTextNormalizer()
    df['hyp'] = df['hyp'].apply(norm)
    df['ref'] = df['ref'].apply(norm)
    df["wer"] = df.apply(lambda row: wer(row['ref'], row['hyp']), axis=1)
    return df

#%%
data_dir = Path.home() / "data/boren/data/ckp/hf_models/phi4_mm_bias_merged/"
json_name = "clean_1000_results.json"
hf_df = load_result(data_dir / f"hf_{json_name}")
vllm_df = load_result(data_dir / f"vllm_{json_name}")
#%%

df = hf_df.merge(vllm_df, on='id', suffixes=('_hf', '_vllm'))

df["werr"]  = df.apply(lambda x: 1 -x['wer_vllm']/(x['wer_hf']+1e-8) , axis=1)

diff_df = df[df['werr'] < -0.05] # WERR less than -5%; regression 

diff_df = diff_df.sort_values(by='werr', ascending=True)
#%%
for index, row in diff_df.iterrows():
    print(f"ID: {row['id']}, WERR: {row['werr']:.2%}, WER HF: {row['wer_hf']:.2%}, WER VLLM: {row['wer_vllm']:.2%}")
    print(f"     REF: {row['ref_hf']}")
    print(f"HF   HYP: {row['hyp_hf']}")
    print(f"VLLM HYP: {row['hyp_vllm']}")
    print("-" * 80)
#%%