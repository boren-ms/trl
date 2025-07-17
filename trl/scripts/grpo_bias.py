# train_grpo.py
# %%
import os
import sys
import pytz
import json
import shortuuid
from pathlib import Path
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
import wandb
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.scripts.audio_dataset import create_audio_dataset
from trl.scripts.audio_metrics import eval_biasing_metrics


def uuid4():
    short_id = shortuuid.ShortUUID().random(length=4)
    return short_id


def init_model(model_id=None, lora_merged=True):
    """Initialize the model and processor."""
    model_id = model_id or "microsoft/Phi-4-multimodal-instruct"
    model_id = model_id.rstrip("/")  # Ensure no trailing slash
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    # lora_adpater=None
    if lora_merged:
        print("Lora merged, delete lora adapters")
        from peft.tuners.lora import LoraLayer

        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.delete_adapter("speech")
                module.delete_adapter("vision")
    else:
        print("loading speech lora adapter")
        model.set_lora_adapter("speech")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


@dataclass
class GRPOScriptArguments:
    """Script arguments for the GRPO training script."""

    job_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the job."},
    )
    project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the project."},
    )
    skip_run_info: bool = field(
        default=False,
        metadata={"help": "Whether to skip to load run info."},
    )
    train_data: Optional[dict] = field(
        default=None,
        metadata={"help": "Training dataset config"},
    )
    eval_data: Optional[dict] = field(
        default=None,
        metadata={"help": "Evalution dataset config"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
    )
    lora_merged: bool = field(
        default=True,
        metadata={"help": "Whether LoRA is merged."},
    )
    reward_funcs: Optional[str] = field(
        default=None,
        metadata={"help": "Reward functions to use. Can be a list of functions or a single function."},
    )


def is_master():
    """Check if the current process is the master process."""
    local_rank = os.environ.get("LOCAL_RANK", "0")
    rank = os.environ.get("RANK", "0")
    print("LocalRank:", local_rank)
    print("Rank:", rank)
    return local_rank == "0" and rank == "0"


def get_job_name(jobname=None):
    """Get a unique job name."""
    if jobname:
        return jobname
    if "--config" in sys.argv:
        # use config file name as job name
        config_file = sys.argv[sys.argv.index("--config") + 1]
        jobname = Path(config_file).stem.split(".")[0]
        return jobname
    # use current time as job name
    tz = pytz.timezone("America/Los_Angeles")  # UTC-7/UTC-8 depending on DST
    return datetime.now(tz).strftime("%Y%m%d-%H%M%S")


def save_run_info(run, work_dir=None, file_name="run_info.json"):
    """Save run identifying information to a JSON file."""
    if work_dir is None:
        return
    info_file = Path(work_dir) / file_name
    info_file.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
    }
    json.dump(info, info_file.open("w"), indent=2)
    print(f"Run info saved to {info_file}")


def load_run_info(work_dir=None, file_name="run_info.json"):
    """Load run info from JSON and resume the run."""
    if work_dir is None:
        return {}
    info_file = Path(work_dir) / file_name
    if not info_file.exists():
        print(f"Run info file {file_name} does not exist in {work_dir}.")
        return {}
    print(f"Loading run info from {info_file}")
    info = json.load(info_file.open("r"))
    url = info.get("run_url", "")
    print(f"Reuse run: {url}")

    parts = Path(url).parts
    if url.startswith("https://msaip.wandb.io/") and len(parts) > 4:
        print("Run info from", url)
        info["run_id"] = info.get("run_id", parts[-1])
        info["project"] = info.get("project", parts[-3])
        info["entity"] = info.get("entity", parts[-4])
    return info


def init_wandb(job_name=None, project=None, config=None, output_dir=None, skip_run_info=False):
    """Initialize wandb."""
    project = os.environ.get("WANDB_PROJECT", project or "biasing")
    job_name = get_job_name(job_name)
    print(f"Project Name: {project}, Run Name: {job_name}")
    key = os.environ.get("WANDB_API_KEY", "")
    host = os.environ.get("WANDB_ORGANIZATION", "")
    wandb.login(host=host, key=key, relogin=True)
    entity = os.environ.get("WANDB_ENTITY", "genai")
    run_info = {} if skip_run_info else load_run_info(output_dir)
    run = wandb.init(
        entity=run_info.get("entity", entity),
        project=run_info.get("project", project),
        id=run_info.get("run_id", None),
        name=run_info.get("run_name", job_name),
        resume="allow",
        config=run_info.get("config", config),
    )
    print("wandb offline: ", run.settings._offline)  # Should be True
    print("wandb mode: ", run.settings.mode)  # Should be "offline"
    save_run_info(run, output_dir)


def reward_functions(names=None):
    """get the reward functions based on the function name."""
    names = names or ["reward_bias_accuracy", "reward_word_accuracy"]
    if isinstance(names, str):
        names = [names]
    funcs = []
    for name in names:
        try:
            module = __import__("trl.scripts.audio_metrics", fromlist=[name])
            funcs.append(getattr(module, name))
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Reward function '{name}' not found.") from e
    return funcs


def create_dataset(config):
    """create dataset"""
    if config is None:
        return None
    if isinstance(config, (list, tuple)):
        datasets = {}
        for i, cfg in enumerate(config):
            nickname = cfg.pop("nickname", f"dataset_{i}")
            datasets[nickname] = create_audio_dataset(**cfg)
        return datasets
    elif isinstance(config, dict):
        return create_audio_dataset(**config)
    raise ValueError("Unsupported dataset config type. Expected dict or list of dicts.")


def main(script_args, training_args):
    """Train the model with GRPO."""
    if is_master():
        print("Init Wandb")
        init_wandb(
            job_name=script_args.job_name,
            project=script_args.project,
            output_dir=training_args.output_dir,
            skip_run_info=script_args.skip_run_info,
        )  # disabled for wandb for orange

    model, processor = init_model(script_args.model_name_or_path, lora_merged=script_args.lora_merged)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions(script_args.reward_funcs),
        args=training_args,
        train_dataset=create_dataset(script_args.train_data),
        eval_dataset=create_dataset(script_args.eval_data),
        processing_class=processor,
        compute_metrics=eval_biasing_metrics,
    )
    print("Training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print("All Done.")


def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create a parser for the GRPO training script."""
    dataclass_types = (GRPOScriptArguments, GRPOConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)

# %%
