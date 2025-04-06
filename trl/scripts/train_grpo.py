# train_grpo.py
# %%
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch
import wandb
from jiwer import process_words
from datetime import datetime
import jiwer.transforms as tr

log_name = f"grpo-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
wandb.init(project="trl-grpo-demo", name=log_name)

# dataset = load_dataset("trl-lib/tldr", split="train")

dataset = load_dataset(
    "hf-audio/esb-datasets-test-only-sorted",
    "librispeech",
    split="test.clean",
)
INSTRUCTION = "Transcribe the clip into text."


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
dataset = dataset.map(process_sample)


def word_error(ref, hyp):
    """Compute the word error rate between two strings."""
    norm = tr.Compose(
        [
            tr.ToLowerCase(),
            tr.ExpandCommonEnglishContractions(),
            tr.SubstituteRegexes({r"\*([^\*]+)\*": r"\1"}),
            tr.RemovePunctuation(),
            tr.RemoveWhiteSpace(replace_by_space=True),
            tr.RemoveMultipleSpaces(),
            tr.Strip(),
            tr.ReduceToSingleSentence(),
            tr.ReduceToListOfListOfWords(),
        ]
    )
    output = process_words(ref, hyp, norm, norm)
    return output.wer * 100

def reward_errors(completions, **kwargs):
    references = kwargs["text"]
    return [-word_error(ref,hyp) for hyp, ref in zip(completions, references)]


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

OUTPUT_DIR = f"output/{MODEL_ID}_GRPO_v0"
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    logging_steps=1,
    # per_device_train_batch_size=2,
    bf16=True,
    gradient_checkpointing=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=["wandb"],
    eval_strategy="no",
    learning_rate=5e-6,
    log_completions=True,
)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_errors,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
)
trainer.train()

# %%
