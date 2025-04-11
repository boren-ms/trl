# %%
from audio_dataset import create_dataset
from omegaconf import OmegaConf
from trl.data_utils import maybe_extract_prompt, maybe_apply_chat_template

yaml_path = "/home/boren/trl/exp_conf/dpo_bias_ls.yaml"
with open(yaml_path, "r") as f:
    conf = OmegaConf.load(f)
    
dataset = create_dataset(
    dataset_name=conf.dataset_name,
    **conf.dataset_config,
)

dataset = dataset.map(maybe_extract_prompt)

print(dataset[0])
print("All done.")

# %%
