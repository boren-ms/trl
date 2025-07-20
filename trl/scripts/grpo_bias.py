# train_grpo.py
# %%
import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoProcessor
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.scripts.audio_dataset import create_audio_dataset
from trl.scripts.audio_metrics import eval_biasing_metrics
from trl.scripts.shared_utils import init_model, is_master, init_wandb, create_dataset, print_modules
from trl.scripts.utils import add_adapter_func


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


def main(script_args, training_args):
    """Train the model with GRPO."""
    print("Init Wandb")
    init_wandb(
        job_name=script_args.job_name,
        project=script_args.project,
        output_dir=training_args.output_dir
    )

    model, processor = init_model(script_args.model_name_or_path, use_grpo=True)
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
