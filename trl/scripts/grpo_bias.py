# train_grpo.py
# %%
import pytz
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
import wandb
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.scripts.audio_dataset import create_dataset


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


@dataclass
class GRPOScriptArguments:
    """Script arguments for the GRPO training script."""

    dataset_name: str = field(metadata={"help": "Dataset name."})
    job_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the script."},
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
    )
    reward_funcs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Can be a list of functions or a single function."
        },
    )


def init_wandb(name=None):
    """Initialize wandb."""
    wandb.login()
    name = name or "grpo-bias"
    tz = pytz.timezone("America/Los_Angeles")  # UTC-7/UTC-8 depending on DST
    log_name = f"{name}-{datetime.now(tz).strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=name, name=log_name)
    return log_name


def reward_functions(names=None):
    """get the reward functions based on the function name."""
    names = names or ["reward_bias_accuracy", "reward_word_accuracy"]
    if isinstance(names, str):
        names = [names]
    funcs = []
    for name in names:
        try:
            module = __import__("trl.scripts.audio_rewards", fromlist=[name])
            funcs.append(getattr(module, name))
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Reward function '{name}' not found.") from e
    return funcs

def main(script_args, training_args):
    """Train the model with GRPO."""
    init_wandb(name=script_args.job_name)
    model, processor = init_model(script_args.model_name_or_path)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions(script_args.reward_funcs),
        args=training_args,
        train_dataset=create_dataset(
            dataset_name=script_args.dataset_name, **script_args.dataset_config
        ),
        processing_class=processor,
    )
    print("Training...")
    trainer.train()
    print("All Done.")


def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create a parser for the GRPO training script."""
    dataclass_types = (GRPOScriptArguments, GRPOConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "grpo", help="Run the GRPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)

# %%
