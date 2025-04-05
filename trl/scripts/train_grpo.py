# train_grpo.py
# %%
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

dataset = load_dataset("trl-lib/tldr", split="train")
# %%

dataset = load_dataset(
    "hf-audio/esb-datasets-test-only-sorted",
    "librispeech",
    split="test.clean",
)
INSTRUCTION = "Transcribe the clip into text."
import torch


def process_sample(sample):
    """Process a sample from the dataset."""
    x = {
        "prompt": [{"role": "user", "content": f"<|audio_1|>{INSTRUCTION}"}],
        "sr": sample["audio"]["sampling_rate"],
        "audio": torch.tensor(sample["audio"]["array"]),
        "completion": sample["text"],
    }
    return x


# dataset = dataset.take(2)
# %%
dataset = dataset.map(process_sample)


# %%
# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


# MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
# MODEL_ID = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)
# model.set_lora_adapter("speech")
# processor = AutoTokenizer.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True,
# )
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

OUTPUT_DIR = f"output/{MODEL_ID}_GRPO"
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    logging_steps=1,
    # per_device_train_batch_size=2,
    bf16=True,
    gradient_checkpointing=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=["tensorboard"],
    eval_strategy="no",
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
)
trainer.train()

# %%
