# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Online DPO training script with bias support for phi4-MM model."""

import os
import sys
import pytz
import json
from pathlib import Path
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForSequenceClassification, AutoTokenizer
import wandb
from trl import OnlineDPOConfig, OnlineDPOTrainer, TrlParser
from trl.scripts.audio_dataset import create_audio_dataset
from trl.scripts.audio_metrics import eval_biasing_metrics

try:
    import shortuuid
    def uuid4():
        """Generate a short UUID."""
        short_id = shortuuid.ShortUUID().random(length=4)
        return short_id
except ImportError:
    import uuid
    def uuid4():
        """Generate a short UUID."""
        return str(uuid.uuid4())[:8]

try:
    from peft.tuners.lora import LoraLayer
except ImportError:
    LoraLayer = None


def init_model(model_id=None, lora_merged=True):
    """Initialize the model and processor for phi4-MM."""
    model_id = model_id or "microsoft/Phi-4-multimodal-instruct"
    model_id = model_id.rstrip("/")  # Ensure no trailing slash
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    
    if lora_merged:
        print("LoRA merged, delete lora adapters")
        if LoraLayer is not None:
            for module in model.modules():
                if isinstance(module, LoraLayer):
                    try:
                        module.delete_adapter("speech")
                        module.delete_adapter("vision")
                    except Exception:
                        pass  # Adapter might not exist
    else:
        print("Loading speech lora adapter")
        if hasattr(model, 'set_lora_adapter'):
            try:
                model.set_lora_adapter("speech")
            except Exception as e:
                print(f"Warning: Could not set speech LoRA adapter: {e}")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


@dataclass
class OnlineDPOScriptArguments:
    """Script arguments for the Online DPO training script."""

    job_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the job."},
    )
    project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the project."},
    )
    train_data: Optional[dict] = field(
        default=None,
        metadata={"help": "Training dataset config"},
    )
    eval_data: Optional[dict] = field(
        default=None,
        metadata={"help": "Evaluation dataset config"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
    )
    lora_merged: bool = field(
        default=True,
        metadata={"help": "Whether LoRA is merged."},
    )
    reward_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model."},
    )
    judge: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the judge to use."},
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


def init_wandb(job_name=None, project=None, config=None, output_dir=None):
    """Initialize wandb."""
    project = os.environ.get("WANDB_PROJECT", project or "biasing")
    job_name = get_job_name(job_name)
    print(f"Project Name: {project}, Run Name: {job_name}")
    key = os.environ.get("WANDB_API_KEY", "")
    host = os.environ.get("WANDB_ORGANIZATION", "")
    wandb.login(host=host, key=key, relogin=True)
    entity = os.environ.get("WANDB_ENTITY", "genai")
    run_info = load_run_info(output_dir)
    run = wandb.init(
        entity=run_info.get("entity", entity),
        project=run_info.get("project", project),
        id=run_info.get("run_id", None),
        name=run_info.get("run_name", job_name),
        resume="allow",
        config=run_info.get("config", config),
    )
    print("wandb offline: ", run.settings._offline)
    print("wandb mode: ", run.settings.mode)
    save_run_info(run, output_dir)


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


def setup_reward_model_and_tokenizer(script_args, model_kwargs):
    """Setup reward model and tokenizer if provided."""
    if script_args.reward_model_path is not None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_path,
            num_labels=1,
            trust_remote_code=True,
            **model_kwargs,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            script_args.reward_model_path,
            trust_remote_code=True,
            truncation=True,
            truncation_side="left",  # since we judge the completion, truncating left is more appropriate
        )
        return reward_model, reward_tokenizer
    return None, None


def setup_judge(script_args):
    """Setup judge if provided."""
    if script_args.judge is not None:
        from trl import HfPairwiseJudge, OpenAIPairwiseJudge, PairRMJudge
        JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}
        judge_cls = JUDGES[script_args.judge]
        return judge_cls()
    return None


def main(script_args, training_args):
    """Train the model with Online DPO."""
    if is_master():
        print("Init Wandb")
        init_wandb(
            job_name=script_args.job_name, 
            project=script_args.project, 
            output_dir=training_args.output_dir
        )

    # Initialize model and processor
    model, processor = init_model(script_args.model_name_or_path, lora_merged=script_args.lora_merged)

    # Setup model kwargs for consistency
    model_kwargs = dict(
        torch_dtype="auto",
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=True,
    )

    # Setup reward model and tokenizer
    reward_model, reward_tokenizer = setup_reward_model_and_tokenizer(script_args, model_kwargs)

    # Setup judge
    judge = setup_judge(script_args)

    # Set processor padding
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    # Create trainer
    trainer = OnlineDPOTrainer(
        model=model,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        train_dataset=create_dataset(script_args.train_data),
        eval_dataset=create_dataset(script_args.eval_data),
        processing_class=processor,
        reward_processing_class=reward_tokenizer,
        compute_metrics=eval_biasing_metrics,
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print("All Done.")


def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create a parser for the Online DPO training script."""
    dataclass_types = (OnlineDPOScriptArguments, OnlineDPOConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "dpo_online_bias", 
            help="Run the Online DPO training script with bias support", 
            dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)