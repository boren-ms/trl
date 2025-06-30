# %%

from transformers import AutoModelForCausalLM
from pathlib import Path
import shutil

# model_path = Path("/home/boren/data/hf_models/Phi-4-multimodal-instruct")
model_path = Path("/home/boren/data/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf")
merged_model_path = model_path.with_name(model_path.name + "_merged")
model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype="auto", trust_remote_code=True,
        _attn_implementation="flash_attention_2").eval()
#%%
from peft.tuners.lora import LoraLayer

for module in model.modules():
    if isinstance(module, LoraLayer):
        module.merge(adapter_names=["speech"])
        
#%%
shutil.copytree(model_path, merged_model_path, dirs_exist_ok=True)
model.save_pretrained(str(merged_model_path), safe_serialization=True)
#%%