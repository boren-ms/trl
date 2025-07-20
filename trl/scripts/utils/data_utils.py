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

"""Data utilities for TRL scripts."""

from trl.scripts.audio_dataset import create_audio_dataset


def create_dataset(config):
    """Create dataset from configuration.
    
    Args:
        config: Dataset configuration. Can be:
            - None: Returns None
            - dict: Single dataset configuration
            - list/tuple: Multiple dataset configurations
    
    Returns:
        Dataset or dictionary of datasets, or None
    """
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