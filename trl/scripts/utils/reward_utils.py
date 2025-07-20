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

"""Reward function utilities for TRL scripts."""


def load_reward_functions(names=None, module_name="trl.scripts.audio_metrics", default_functions=None):
    """Load reward functions by name from a specified module.
    
    Args:
        names: Function name(s) to load. Can be a string or list of strings.
        module_name: Module to import functions from.
        default_functions: Default function names to use if names is None.
    
    Returns:
        List of reward function objects
    
    Raises:
        ValueError: If a requested function is not found in the module.
    """
    default_functions = default_functions or ["reward_bias_accuracy", "reward_word_accuracy"]
    names = names or default_functions
    if isinstance(names, str):
        names = [names]
    
    funcs = []
    for name in names:
        try:
            module = __import__(module_name, fromlist=[name])
            funcs.append(getattr(module, name))
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Reward function '{name}' not found in module '{module_name}'.") from e
    return funcs