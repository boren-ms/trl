# train_grpo.py
# %%
from pathlib import Path
from safetensors import safe_open

#%%
# root_dir = "/datablob1"
root_dir = "/mnt2/newhome/boren/blobfuse/highperf01eus_data"
# root_dir = "/mnt2/newhome/boren/blobfuse/tsstd01uks_data"
model_id = f"{root_dir}/users/boren/data/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf/5000_hf"
model_id = f"{root_dir}/users/boren/data/hf_models/phi-libri_ft_m1000_p8_new-QpHq_v1"
# model_id = "/home/boren/data/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf"
# model_id = f"{root_dir}/users/boren/data/hf_models/phi4_mm_bias"
model_dir = Path(model_id)
model_file = model_dir / "model-00001-of-00003.safetensors"

with safe_open(model_file, framework="pt", device="cpu") as f:
    metadata = f.metadata()
    print(metadata)
# %%
