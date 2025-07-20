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

"""Model utilities for TRL scripts."""

import shortuuid
from transformers import AutoModelForCausalLM, AutoProcessor
from trl.scripts.utils import add_adapter_func, human_readable


def uuid4():
    short_id = shortuuid.ShortUUID().random(length=4)
    return short_id


def init_model(model_id=None):
    """Initialize the model and processor."""
    model_id = model_id or "microsoft/Phi-4-multimodal-instruct"
    model_id = model_id.rstrip("/")  # Ensure no trailing slash
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    )
    model.set_lora_adapter("speech")
    model = add_adapter_func(model)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def print_modules(model, trainable=True):
    """List trainable modules in the model and total trainable parameter size."""
    print(f"List modules in the model:", {model.__class__.__name__})
    n_total = 0
    n_trainable = 0
    for name, param in model.named_parameters():
        n_total += param.numel()
        if trainable and param.requires_grad:
            print(f"{name}: {human_readable(param.numel())} trainable")
            n_trainable += param.numel()
    print(f"Total trainable: {human_readable(n_trainable)}")
    print(f"Total parameter: {human_readable(n_total)}")
    return n_total, n_trainable