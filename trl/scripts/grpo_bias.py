# train_grpo.py
# %%
import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.scripts.audio_dataset import create_audio_dataset
from trl.scripts.audio_metrics import eval_biasing_metrics
from trl.scripts.shared_utils import init_model, is_master, init_wandb, create_dataset
from trl.scripts.utils import add_adapter_func, human_readable


def init_model_grpo(model_id=None):
    """Initialize the model and processor with GRPO-specific settings."""
    model, processor = init_model(model_id, lora_merged=False)
    model = add_adapter_func(model)
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
    reward_funcs: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Can be a list of functions or a single function."
        },
    )


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


def init_wandb_grpo(job_name=None, project=None, config=None, output_dir=None, skip_run_info=False):
    """Initialize wandb with GRPO-specific skip_run_info support."""
    from trl.scripts.shared_utils import get_job_name, load_run_info, save_run_info
    import os
    import wandb
    
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


def print_modules(model, trainable=True):
    """List trainable modules in the model and total trainable parameter size."""
    print(f"List modules in the model:", {model.__class__.__name__})
    n_total = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if trainable and param.requires_grad:
            print(f"{name}: {human_readable(param.numel())} trainable")
            n_trainable += param.numel()
    print(f"Total trainable: {human_readable(n_trainable)}")
    print(f"Total parameter: {human_readable(n_total)}")
    return n_total, n_trainable


def main(script_args, training_args):
    """Train the model with GRPO."""
    if is_master():
        print("Init Wandb")
        # Use custom init_wandb with skip_run_info support
        init_wandb_grpo(
            job_name=script_args.job_name,
            project=script_args.project,
            output_dir=training_args.output_dir,
            skip_run_info=script_args.skip_run_info,
        )

    model, processor = init_model_grpo(script_args.model_name_or_path)
    _, n_trainable = print_modules(model, trainable=True)
    assert n_trainable > 0, "No trainable parameters found in the model."

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
