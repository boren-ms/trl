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

"""Argument parsing utilities for TRL scripts."""

import argparse
from trl import TrlParser


def make_parser(subparsers=None, dataclass_types=None, parser_name=None, help_text=None):
    """Create a parser for TRL training scripts.
    
    Args:
        subparsers: Optional subparsers action for adding as a subcommand.
        dataclass_types: Tuple of dataclass types to use for argument parsing.
        parser_name: Name for the subparser (required if subparsers provided).
        help_text: Help text for the subparser (optional).
    
    Returns:
        TrlParser instance
    """
    if subparsers is not None:
        if not parser_name:
            raise ValueError("parser_name is required when using subparsers")
        parser = subparsers.add_parser(
            parser_name, 
            help=help_text or f"Run the {parser_name.upper()} training script", 
            dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser