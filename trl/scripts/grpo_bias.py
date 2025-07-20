# train_grpo.py
# %%
from trl import GRPOTrainer
from trl.scripts.audio_metrics import eval_biasing_metrics
from trl.scripts.utils.argument_utils import GRPOScriptArguments, make_parser
from trl.scripts.utils.data_utils import create_dataset
from trl.scripts.utils.job_utils import is_master
from trl.scripts.utils.model_utils import init_model, print_modules
from trl.scripts.utils.reward_utils import reward_functions
from trl.scripts.utils.wandb_utils import init_wandb


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

    model, processor = init_model(script_args.model_name_or_path)
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


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)

# %%
