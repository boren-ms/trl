#%%
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

training_args = GRPOConfig(
    output_dir="my_test",
    use_vllm=True,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B",
    args=training_args,
    reward_funcs=reward_num_unique_chars,
    train_dataset=dataset,
)

trainer.train()
#%%