# %%
# train_online_dpo.py
from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

import os

os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_ENTITY"] = "genai"
# %%
model_path = f"{Path.home()}/data/ckp/hf_models/Qwen2.5-0.5B-Instruct"
reward_model_path = f"{Path.home()}/data/ckp/hf_models/Qwen2-0.5B-Reward"
# %%
data_path = f"{Path.home()}/data/gsm8k"
dataset = load_dataset(str(data_path), data_files={"train": "train.parquet", "test": "test.parquet"}, split="train")
# %%
reward_model = AutoModelForSequenceClassification.from_pretrained((reward_model_path), num_labels=1)
reward_tokenizer = AutoTokenizer.from_pretrained((reward_model_path))
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# %%
training_args = OnlineDPOConfig(output_dir="Qwen2-0.5B-OnlineDPO", per_device_train_batch_size=2)  # Reduce batch size to avoid OOM
trainer = OnlineDPOTrainer(
    model=model,
    reward_model=reward_model,
    reward_processing_class=reward_tokenizer,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()
# %%
