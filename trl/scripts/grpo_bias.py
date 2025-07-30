# train_grpo.py
# %%
import argparse
from functools import partial
from dataclasses import dataclass, field
from typing import Optional
from trl import GRPOConfig, GRPOTrainer, TrlParser
from trl.scripts.audio_metrics import eval_biasing_metrics
from trl.scripts.shared_utils import init_model, WandbHelper, create_dataset, print_modules, get_latest_valid_checkpoint


@dataclass
class GRPOScriptArguments:
    """Script arguments for the GRPO training script."""

    new_run: bool = field(
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
    new_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use a new LoRA adapter"},
    )
    reward_funcs: Optional[str] = field(
        default=None,
        metadata={"help": "Reward functions to use. Can be a list of functions or a single function."},
    )
    reward_func_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "Keyword arguments for the reward functions."},
    )


def reward_functions(names=None, **kwargs):
    """get the reward functions based on the function name."""
    names = names or ["reward_bias_accuracy", "reward_word_accuracy"]
    if isinstance(names, str):
        names = [names]
    funcs = []
    for name in names:
        try:
            module = __import__("trl.scripts.audio_metrics", fromlist=[name])
            func = getattr(module, name)
            if kwargs:
                func = partial(func, **kwargs)
            funcs.append(func)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Reward function '{name}' not found.") from e
    return funcs


def main(script_args, training_args):
    """Train the model with GRPO."""
    WandbHelper(
        work_dir=training_args.output_dir,
        new_run=script_args.new_run,
    ).init(main_only=True)

    lora_name = "speech" if script_args.new_lora else None
    model, processor = init_model(script_args.model_name_or_path, new_lora=lora_name)
    _, n_trainable = print_modules(model)
    assert n_trainable > 0, "No trainable parameters found in the model."

    reward_func_kwargs = script_args.reward_func_kwargs or {}
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions(script_args.reward_funcs, **reward_func_kwargs),
        args=training_args,
        train_dataset=create_dataset(script_args.train_data),
        eval_dataset=create_dataset(script_args.eval_data),
        processing_class=processor,
        compute_metrics=eval_biasing_metrics,
    )
    print("Training...")
    latest_chkp_dir = get_latest_valid_checkpoint(training_args.output_dir)
    if latest_chkp_dir:
        print("Resuming from ", latest_chkp_dir)
    trainer.train(resume_from_checkpoint=latest_chkp_dir)
    trainer.save_model()
    print("Training completed.")


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
