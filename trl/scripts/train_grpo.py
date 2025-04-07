# train_grpo.py
# %%
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoProcessor
import wandb
from jiwer import process_words
from datetime import datetime
import jiwer.transforms as tr
from audio_set import create_dataset

log_name = f"grpo-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
wandb.init(project="trl-grpo-demo", name=log_name)


dataset = create_dataset(name="bias", num=200)
# dataset = create_dataset(name="ls", num=200)
# dataset = dataset.map(process_sample)

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
    # print(f"ref: {ref}")
    # print(f"hyp: {hyp}")
    output = process_words(ref, hyp, norm, norm)
    return output.wer * 100


def reward_errors(completions, **kwargs):
    """Compute the reward for a list of completions."""
    references = kwargs["text"]
    return [
        -word_error(ref, completion[-1]["content"])
        for completion, ref in zip(completions, references)
    ]


# MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
# MODEL_ID = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)
model.set_lora_adapter("speech")

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

OUTPUT_DIR = f"output/{MODEL_ID}_GRPO_v0"
NUM_SAMPLE=4
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    logging_steps=1,
    bf16=True,
    gradient_checkpointing=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=["wandb"],
    eval_strategy="no",
    learning_rate=5e-6,
    log_completions=True,
    max_prompt_length=1024,
    max_completion_length=512,
    per_device_train_batch_size=NUM_SAMPLE,
    num_generations=NUM_SAMPLE,
    wandb_log_unique_prompts=True,
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
