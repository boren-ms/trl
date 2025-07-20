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

import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import OnlineDPOConfig, OnlineDPOTrainer, TrlParser
from trl.scripts.audio_metrics import eval_biasing_metrics
from trl.scripts.shared_utils import (
    init_model, is_master, get_job_name, save_run_info, load_run_info,
    init_wandb, create_dataset
)


def setup_reward_model_and_tokenizer(reward_model_path, model_kwargs):
    """Setup reward model and tokenizer if provided."""
    if reward_model_path is not None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            num_labels=1,
            trust_remote_code=True,
            **model_kwargs,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_model_path,
            trust_remote_code=True,
            truncation=True,
            truncation_side="left",  # since we judge the completion, truncating left is more appropriate
        )
        return reward_model, reward_tokenizer
    return None, None


def setup_judge(judge_name):
    """Setup judge if provided."""
    if judge_name is not None:
        from trl import HfPairwiseJudge, OpenAIPairwiseJudge, PairRMJudge
        JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}
        judge_cls = JUDGES[judge_name]
        return judge_cls()
    return None


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

    reward_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model."},
    )
    judge: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the judge to use."},
    )


def main(script_args, training_args):
    """Train the model with Online DPO."""
    print("Init Wandb")
    init_wandb(
        job_name=script_args.job_name, 
        project=script_args.project, 
        output_dir=training_args.output_dir
    )

    # Initialize model and processor
    model, processor = init_model(script_args.model_name_or_path)

    # Setup model kwargs for consistency
    model_kwargs = dict(
        torch_dtype="auto",
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=True,
    )

    # Setup reward model and tokenizer
    reward_model, reward_tokenizer = setup_reward_model_and_tokenizer(script_args.reward_model_path, model_kwargs)

    # Setup judge
    judge = setup_judge(script_args.judge)

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