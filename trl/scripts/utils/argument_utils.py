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

"""Argument utilities for TRL scripts."""

import argparse
from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig, TrlParser


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
        metadata={"help": "Reward functions to use. Can be a list of functions or a single function."},
    )


def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create a parser for the GRPO training script."""
    dataclass_types = (GRPOScriptArguments, GRPOConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser