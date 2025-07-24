"""Shared utility functions for bias training scripts."""

# %%
import os
import sys
import pytz
import json
import wandb
import blobfile as bf
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState
from trl.trainer.utils import add_adapter_func
from trl.scripts.audio_dataset import create_audio_dataset


def get_speech_peft_model(model, lora_name):
    config = model.config
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.speech_lora["r"],
        lora_alpha=config.speech_lora["lora_alpha"],
        target_modules=config.speech_lora["layer"],
        lora_dropout=config.speech_lora["dp"],
        task_type="CAUSAL_LM",
    )
    get_peft_model(model.model, lora_config, adapter_name=lora_name)
    return model


def init_model(model_id=None, new_lora=None):
    """Initialize the model and processor."""
    model_id = model_id or "microsoft/Phi-4-multimodal-instruct"
    model_id = model_id.rstrip("/")  # Ensure no trailing slash
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    model.set_lora_adapter("speech")
    model = add_adapter_func(model)
    if new_lora:
        print("merge and unload model")
        model.merge_and_unload()  # merge lora and back to normal Linear
        print("Prepare peft model with adapter:", new_lora)
        model = get_speech_peft_model(model, lora_name=new_lora)  # revert peft model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


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
    work_dir = work_dir or Path.cwd()
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
    work_dir = work_dir or Path.cwd()
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


def print_modules(model, trainable=False):
    """List trainable modules in the model and total trainable parameter size."""
    print(f"List modules in the model:", {model.__class__.__name__})
    n_total = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if param.requires_grad:
            n_trainable += param.numel()
            if trainable:
                print(f"{name}: {human_readable(param.numel())} trainable")
    print(f"Total trainable: {human_readable(n_trainable)}")
    print(f"Total parameter: {human_readable(n_total)}")
    return n_total, n_trainable


def init_wandb(job_name=None, project=None, config=None, output_dir=None, skip_run_info=False):
    """Initialize wandb."""
    if not PartialState().is_main_process:
        return None

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
    return run


def create_dataset(config):
    """Create dataset."""
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


def human_readable(num):
    """Convert a number to human readable format (K, M, G)."""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}G"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)


def is_valid_checkpoint(model_dir):
    """Check if the model path is valid."""
    if not bf.exists(model_dir):
        # print(f"Model path {model_dir} does not exist.")
        return False
    if not bf.isdir(model_dir):
        # print(f"Model path {model_dir} is not a directory.")
        return False
    config_file = f"{model_dir}/config.json"
    if not bf.exists(config_file):
        # print(f"Config file {config_file} does not exist in the model directory.")
        return False
    if not any(bf.glob(f"{model_dir}/*.safetensors")):
        # print(f"No .safetensors files found in {model_dir}.")
        return False
    return True


def get_latest_valid_checkpoint(output_dir):
    """Get the latest valid checkpoint directory."""
    latest_chkp_dir = get_last_checkpoint(output_dir)
    if is_valid_checkpoint(latest_chkp_dir):
        return latest_chkp_dir
    return None
