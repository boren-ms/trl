"""DPO training script."""
import pytz
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
import wandb
from trl import DPOConfig, DPOTrainer, TrlParser
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
class DPOScriptArguments:
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


def init_wandb(name=None):
    """Initialize wandb."""
    wandb.login()
    name = name or "dpo-bias"
    tz = pytz.timezone("America/Los_Angeles")  # UTC-7/UTC-8 depending on DST
    log_name = f"{name}-{datetime.now(tz).strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=name, name=log_name)
    return log_name


def main(script_args, training_args):
    """Train the model with DPO."""
    init_wandb(name=script_args.job_name)
    model, processor = init_model(script_args.model_name_or_path)

    trainer = DPOTrainer(
        model,
        None,
        args=training_args,
        train_dataset=create_dataset(
            dataset_name=script_args.dataset_name, **script_args.dataset_config
        ),
        eval_dataset=None,
        processing_class=processor,
    )
    print("Training...")
    trainer.train()
    print("All Done.")


def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create the argument parser for the DPO script."""
    dataclass_types = (DPOScriptArguments, DPOConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo", help="Run the DPO training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)
