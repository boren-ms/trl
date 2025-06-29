# %%
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

# dataset = load_dataset("trl-lib/tldr", split="train")

data_path = "/home/boren/data/gsm8k"
dataset = load_dataset(
    data_path,
    data_files={"train": "train.parquet", "test": "test.parquet"},
    split="train",
)

model_path = "/home/boren/data//ckp/Qwen/models--Qwen--Qwen2.5-0.5B-Instruct"


# model_path="Qwen/Qwen2.5-7B",
# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    """Reward function that counts the number of unique characters in the completions."""
    return [len(set(c)) for c in completions]


training_args = GRPOConfig(
    output_dir="my_test",
    use_vllm=True,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model=model_path,
    args=training_args,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
)

trainer.train()
# %%
