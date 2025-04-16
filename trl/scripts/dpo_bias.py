"""DPO training script."""

import pytz
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
import wandb
from trl import DPOConfig, DPOTrainer, TrlParser, ModelConfig, get_peft_config
from trl.scripts.audio_dataset import create_dataset

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer



def clone_phimm_lora(model,  dst_name, src_name="speech"):
    """Clone the LoRA adapter from src_name to dst_name."""
    print(f"Cloning LoRA adapter from {src_name} to {dst_name}")
    src_lora_config = getattr(model.config, f"{src_name}_lora")
    lora_conf = LoraConfig(
        r=src_lora_config['r'],
        lora_alpha=src_lora_config['lora_alpha'],
        target_modules=src_lora_config['layer'],
        lora_dropout=src_lora_config['dp'],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model.model, lora_conf, adapter_name=dst_name)
    for module in peft_model.modules():
        if not isinstance(module, LoraLayer):
            continue
        if module.merged:
            module.unmerge()
        module.lora_A[dst_name].weight.data = module.lora_A[src_name].weight.data
        module.lora_B[dst_name].weight.data = module.lora_B[src_name].weight.data
        # module.set_adapter(dst_name)
        # module._disable_adapters = False
    return peft_model

def init_model(model_id=None, ref_adapter=None):
    """Initialize the model and processor."""
    model_id = model_id or "microsoft/Phi-4-multimodal-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    if ref_adapter is not None:
        model = clone_phimm_lora(model, ref_adapter, "speech") # PEFT model
        model.set_adapter("speech")
    else:
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
    model, processor = init_model(script_args.model_name_or_path, training_args.ref_adapter_name)
    if training_args.ref_adapter_name is not None:
        training_args.model_apdapter_name = "speech"
    trainer = DPOTrainer(
        model,
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
