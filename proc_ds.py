# %%
from trl.scripts.audio_dataset import create_audio_dataset
import yaml
from pathlib import Path

# %%
conf_path = Path("/mnt2/newhome/boren/trl/orng_conf/biasing/data/hcv2_sc3k_fr01.yaml")

conf = yaml.safe_load(conf_path.read_text())["train_data"]
conf.update({"cache_name": conf_path.stem})
print("Config: ", conf)

ds = create_audio_dataset(**conf)
print("Got dataset info")
print(ds)
print("dataset samples:")
egs = ds[0]
n_words = len(egs["text"].split())
n_keywords = len(egs["keywords"])
print("text:", egs["text"])
print("keywords:", egs["keywords"])
print(f"num_words: {n_words}, num_keywords: {n_keywords}, ratio: {n_keywords/(n_words+1e-6):.3f}")

# %%
