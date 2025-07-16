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

"""
DPO Online Training for Phi4-MultiModal Model with Audio and Text Prompts

Example usage:
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --dataset_name openasr \
    --learning_rate 5.0e-7 \
    --output_dir phi4-multimodal-online-dpo \
    --use_peft
"""

import torch
from transformers import AutoProcessor

from trl import (
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.scripts.audio_utils import init_model
from trl.scripts.audio_dataset import create_audio_dataset


def create_multimodal_dataset(script_args):
    """Create a simple dataset for multimodal training with audio and text."""
    dataset_config = {
        "dataset_name": "openasr",
        "name": "librispeech",
        "split": "test.clean",
        "num_egs": 1000,  # Fixed reasonable size
        "streaming": False,
        "biasing": {
            "bias_prob": 0.9,
            "hit_prob": 0.9,
            "max_piece_len": 1,
            "max_num": 2,
        },
        "simu_perference": {
            "error_range": (0.1, 0.25),
            "delete_prob": 0.1,
            "substitute_prob": 0.1,
            "insert_prob": 0.05,
        },
        "load_audio": True,
    }
    return create_audio_dataset(**dataset_config)


def main():
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Set gradient checkpointing for memory efficiency
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Initialize the phi4-multimodal model and processor
    print("Initializing Phi4-MultiModal model...")
    model, processor = init_model(
        model_id=model_args.model_name_or_path or "microsoft/Phi-4-multimodal-instruct",
        lora_merged=not getattr(model_args, 'use_peft', False)
    )
    
    # Set up model configuration for training
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    
    # Configure device mapping for quantization if needed
    if quantization_config is not None:
        model.to(device=get_kbit_device_map())

    # Set up chat template
    tokenizer = processor.tokenizer
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create multimodal dataset
    print("Creating multimodal dataset...")
    dataset = create_multimodal_dataset(script_args)

    # Initialize the OnlineDPOTrainer
    print("Initializing OnlineDPOTrainer...")
    trainer = OnlineDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )

    # Start training
    print("Starting DPO online training...")
    trainer.train()

    # Save the trained model
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()