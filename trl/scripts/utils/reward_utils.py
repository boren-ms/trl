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

"""Reward utilities for TRL scripts."""


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