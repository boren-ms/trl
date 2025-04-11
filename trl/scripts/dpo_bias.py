# Copyright 2025 The HuggingFace Team. All rights reserved.
#
"""DPO training script."""

import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from trl import DPOConfig, DPOTrainer, TrlParser
from trl.scripts.audio_dataset import create_dataset


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


def main(script_args, training_args):
    """Main function for the DPO training script."""
    
    dataset = create_dataset(
        dataset_name=script_args.dataset_name,
        **script_args.dataset_config,
    )
    model, processor = init_model()

    trainer = DPOTrainer(
        model,
        None,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        processing_class=processor,
        # peft_config=peft_config,
    )
    print("Training...")
    trainer.train()
    print("Training complete.")


def make_parser(subparsers: argparse._SubParsersAction = None):
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
