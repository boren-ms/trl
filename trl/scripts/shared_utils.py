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
from trl.trainer.utils import add_adapter_func, rank_print
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
        rank_print("merge and unload model")
        model.merge_and_unload()  # merge lora and back to normal Linear
        rank_print("Prepare peft model with adapter:", new_lora)
        model = get_speech_peft_model(model, lora_name=new_lora)  # revert peft model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def get_config_path(config_path=None):
    """Get the config path from command line arguments or environment variables."""
    if config_path:
        return Path(config_path).resolve()
    if "--config" in sys.argv:
        config_index = sys.argv.index("--config") + 1
        if config_index < len(sys.argv):
            return Path(sys.argv[config_index]).resolve()
    return None


class WandbHelper:
    """Helper class to manage wandb initialization and run info."""

    def __init__(self, run_name=None, work_dir=None, new_run=False):
        self.new_run = new_run
        self.run_name = run_name
        work_dir = work_dir or Path.cwd()
        self.run_info_file = Path(work_dir) / "run_info.json"

    def _wandb_info(self):
        """Get wandb information from environment variables."""
        config_path = get_config_path()
        if not config_path:
            run_name = "default_job_name"
            project = os.environ.get("WANDB_PROJECT", "biasing")
        else:
            run_name = config_path.stem
            project = config_path.parent.name or "biasing"
        run_name = self.run_name or run_name
        run_info = {} if self.new_run else self._load_info()

        return {
            "entity": run_info.get("entity", "genai"),
            "project": run_info.get("project", project),
            "name": run_info.get("run_name", run_name),
            "id": run_info.get("run_id", None),
            "config": run_info.get("config", {}),
        }

    def _login(self):
        """Login to wandb."""
        key = os.environ.get("WANDB_API_KEY", "")
        host = os.environ.get("WANDB_ORGANIZATION", "https://msaip.wandb.io")
        wandb.login(host=host, key=key, relogin=True)

    def init(self, main_only=True):
        """Initialize wandb."""
        if not PartialState().is_main_process and main_only:
            return None
        self._login()
        run_info = self._wandb_info()
        run = wandb.init(**run_info, resume="allow")
        rank_print("wandb mode: ", run.settings.mode)  # Should be "offline"
        self.run_info_file = self._save_info(run)
        return run

    def _save_info(self, run):
        """Save run identifying information to a JSON file."""
        self.run_info_file.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "entity": run.entity,
            "project": run.project,
            "run_id": run.id,
            "run_name": run.name,
            "run_url": run.url,
        }
        json.dump(info, self.run_info_file.open("w"), indent=2)
        rank_print(f"Run info saved to {self.run_info_file}")
        return self.run_info_file

    def _load_info(self):
        """Load run info from JSON and resume the run."""
        if not self.run_info_file.exists():
            rank_print(f"Run info file {self.run_info_file} does not exist.")
            return {}
        rank_print(f"Loading run info from {self.run_info_file}")
        info = json.load(self.run_info_file.open("r"))
        url = info.get("run_url", None)
        if not url:
            return info
        rank_print(f"Reuse run: {url}")
        parts = Path(url).parts
        if url.startswith("https://msaip.wandb.io/") and len(parts) > 4:
            rank_print("Run info from", url)
            info["run_id"] = info.get("run_id", parts[-1])
            info["project"] = info.get("project", parts[-3])
            info["entity"] = info.get("entity", parts[-4])
        return info


def print_modules(model, trainable=False):
    """List trainable modules in the model and total trainable parameter size."""
    rank_print("List modules in the model:", {model.__class__.__name__})
    n_total = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if param.requires_grad:
            n_trainable += param.numel()
            if trainable:
                rank_print(f"{name}: {human_readable(param.numel())} trainable")
    rank_print(f"Total trainable: {human_readable(n_trainable)}")
    rank_print(f"Total parameter: {human_readable(n_total)}")
    return n_total, n_trainable


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
    if not model_dir:
        return False
    if not bf.exists(model_dir):
        # rank_print(f"Model path {model_dir} does not exist.")
        return False
    if not bf.isdir(model_dir):
        # rank_print(f"Model path {model_dir} is not a directory.")
        return False
    config_file = f"{model_dir}/config.json"
    if not bf.exists(config_file):
        # rank_print(f"Config file {config_file} does not exist in the model directory.")
        return False
    if not any(bf.glob(f"{model_dir}/*.safetensors")):
        # rank_print(f"No .safetensors files found in {model_dir}.")
        return False
    return True


def get_latest_valid_checkpoint(output_dir):
    """Get the latest valid checkpoint directory."""
    latest_chkp_dir = get_last_checkpoint(output_dir)
    if is_valid_checkpoint(latest_chkp_dir):
        return latest_chkp_dir
    return None
