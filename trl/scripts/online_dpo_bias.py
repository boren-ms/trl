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
from trl import OnlineDPOConfig, OnlineDPOTrainer, TrlParser
from trl.scripts.audio_metrics import eval_biasing_metrics, compute_wers
from trl.scripts.shared_utils import init_model, WandbHelper, create_dataset, print_modules
from trl import BasePairwiseJudge


def init_reward_model(reward_path, **kwargs):
    """Setup reward model and tokenizer if provided."""
    if not reward_path:
        return None, None
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_path,
        num_labels=1,
        trust_remote_code=True,
        **kwargs,
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(
        reward_path,
        trust_remote_code=True,
        truncation=True,
        truncation_side="left",  # since we judge the completion, truncating left is more appropriate
    )
    return reward_model, reward_tokenizer


class ErrorJudge(BasePairwiseJudge):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def judge(self, prompts, completions, shuffle_order=False):
        return [0 if self.calc_wer(ref, hyps[0]) <= self.calc_wer(ref, hyps[1]) else 1 for ref, hyps in zip(prompts, completions)]

    def calc_wer(self, ref, hyp):
        wer, uwer, bwer = compute_wers([{"ref": ref, "hyp": hyp}], **self.kwargs)
        return wer.get_wer()


def setup_judge(judge_name, **kwargs):
    """Setup judge if provided."""
    if not judge_name:
        return
    from trl import HfPairwiseJudge, OpenAIPairwiseJudge, PairRMJudge

    JUDGES = {
        "pair_rm": PairRMJudge,
        "openai": OpenAIPairwiseJudge,
        "hf": HfPairwiseJudge,
        "error": ErrorJudge,
    }
    judge_cls = JUDGES[judge_name]
    return judge_cls(**kwargs)


@dataclass
class OnlineDPOScriptArguments:
    """Script arguments for the Online DPO training script."""

    new_run: bool = field(
        default=False,
        metadata={"help": "whether skip the run info from checkpoint"},
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
    judge_kwargs: Optional[dict] = field(
        default=None,
        metadata={"help": "Additional arguments for the judge."},
    )


def main(script_args, training_args):
    """Train the model with Online DPO."""
    WandbHelper(
        work_dir=training_args.output_dir,
        new_run=script_args.new_run,
    ).init(main_only=True)

    # Initialize model and processor
    model, processor = init_model(script_args.model_name_or_path, new_lora="speech")  # use same speech lora name
    _, n_trainable = print_modules(model)
    assert n_trainable > 0, "No trainable parameters found in the model."
    # Setup reward model and tokenizer
    reward_model, reward_tokenizer = init_reward_model(training_args.reward_model_path)

    # Setup judge
    judge_kwargs = script_args.judge_kwargs or {}
    judge = setup_judge(training_args.judge, **judge_kwargs)

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
        # compute_metrics=eval_biasing_metrics,
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
            dataclass_types=dataclass_types,
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)
