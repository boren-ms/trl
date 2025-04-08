# train_grpo.py
# %%
from datetime import datetime
from jiwer import process_words
import jiwer.transforms as tr
from trl.scripts.audio_dataset import create_dataset
import wandb
from transformers import AutoModelForCausalLM, AutoProcessor
from trl import GRPOConfig, GRPOTrainer


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


def init_model(model_id=None):
    """Initialize the model and processor."""
    model_id = model_id or "microsoft/Phi-4-multimodal-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    model.set_lora_adapter("speech")
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return model, processor


def grpo_train(
    name="phi4_mm_grpo_ls",
    output_dir=None,
    dataset="ls",
    batch_size=None,
    num_sample=4,
    lr=5e-6,
):
    """Train the model with GRPO."""
    wandb.login()
    log_name = f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=f"{name}", name=log_name)
    batch_size = batch_size or num_sample
    if batch_size % num_sample != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by num_sample {num_sample}"
        )
    output_dir = output_dir or f"output/{log_name}"
    print(f"Dataset: {dataset}")
    print(f"Output dir: {output_dir}")
    model, processor = init_model()
    training_args = GRPOConfig(
        output_dir=output_dir,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        logging_dir=f"{output_dir}/logs",
        report_to=["wandb"],
        eval_strategy="no",
        learning_rate=lr,
        log_completions=True,
        max_prompt_length=1024,
        max_completion_length=512,
        per_device_train_batch_size=num_sample,
        num_generations=num_sample,
        wandb_log_unique_prompts=True,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_errors,
        args=training_args,
        train_dataset=create_dataset(name=dataset),
        processing_class=processor,
    )
    print("Training...")
    trainer.train()
    print("All Done.")


if __name__ == "__main__":
    import fire

    fire.Fire(grpo_train)
