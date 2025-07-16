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
DPO Online Training Example for Phi4-MultiModal Model with Audio and Text Prompts

This script demonstrates how to use online DPO (Direct Preference Optimization) training
with Microsoft's Phi4-MultiModal model, handling both audio and text inputs.

Usage:
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --dataset_name openasr \
    --learning_rate 5.0e-7 \
    --output_dir phi4-multimodal-online-dpo \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --missing_eos_penalty 1.0 \
    --max_new_tokens 128 \
    --use_peft \
    --lora_target_modules=all-linear

With custom audio dataset:
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --dataset_name tsv \
    --dataset_config_path audio_dataset_config.json \
    --learning_rate 5.0e-7 \
    --output_dir phi4-multimodal-online-dpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --use_peft
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from trl import (
    HfPairwiseJudge,
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    OpenAIPairwiseJudge,
    PairRMJudge,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from trl.scripts.audio_utils import init_model
from trl.scripts.audio_dataset import create_audio_dataset


@dataclass
class MultimodalScriptArguments(ScriptArguments):
    """Extended script arguments for multimodal training."""
    
    dataset_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dataset configuration JSON file for custom dataset setup."}
    )
    tsv_paths: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of TSV file paths for TSV dataset type."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples to use."}
    )


# Available judges for preference evaluation
JUDGES = {"pair_rm": PairRMJudge, "openai": OpenAIPairwiseJudge, "hf": HfPairwiseJudge}


def create_multimodal_dataset(script_args, training_args):
    """Create a dataset for multimodal training with audio and text."""
    if hasattr(script_args, 'dataset_config_path') and script_args.dataset_config_path:
        # Load dataset configuration from file
        import json
        with open(script_args.dataset_config_path, 'r') as f:
            dataset_config = json.load(f)
        dataset = create_audio_dataset(**dataset_config)
    else:
        # Use default configuration for the specified dataset
        if script_args.dataset_name == "openasr":
            dataset_config = {
                "dataset_name": "openasr",
                "name": "librispeech",
                "split": "test.clean",
                "num_egs": getattr(script_args, 'max_train_samples', None),
                "streaming": getattr(script_args, 'dataset_streaming', False),
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
        elif script_args.dataset_name == "tsv":
            # Example TSV configuration - users should provide their own TSV files
            dataset_config = {
                "dataset_name": "tsv",
                "tsv_paths": getattr(script_args, 'tsv_paths', []),
                "num_egs": getattr(script_args, 'max_train_samples', None),
                "streaming": getattr(script_args, 'dataset_streaming', False),
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
        else:
            raise ValueError(f"Unsupported dataset name: {script_args.dataset_name}")
        
        dataset = create_audio_dataset(**dataset_config)
    
    return dataset


def main():
    parser = TrlParser((MultimodalScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Set gradient checkpointing for memory efficiency
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # Initialize the phi4-multimodal model and processor
    print("Initializing Phi4-MultiModal model...")
    model, processor = init_model(
        model_id=model_args.model_name_or_path or "microsoft/Phi-4-multimodal-instruct",
        lora_merged=not getattr(model_args, 'use_peft', False)
    )
    
    # Set up additional model configuration for training
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    
    # Configure device mapping for quantization if needed
    if quantization_config is not None:
        model.to(device=get_kbit_device_map())

    # Initialize reward model if specified
    reward_model = None
    reward_processor = None
    if training_args.reward_model_path is not None:
        # For multimodal models, we assume the reward model can also handle multimodal inputs
        print(f"Loading reward model from {training_args.reward_model_path}")
        reward_model = AutoModelForCausalLM.from_pretrained(
            training_args.reward_model_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        reward_processor = AutoProcessor.from_pretrained(
            training_args.reward_model_path,
            trust_remote_code=model_args.trust_remote_code,
        )

    # Initialize judge if specified
    judge = None
    if training_args.judge is not None:
        judge_cls = JUDGES[training_args.judge]
        judge = judge_cls()

    # Set up chat template if not present
    tokenizer = processor.tokenizer
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create multimodal dataset
    print("Creating multimodal dataset...")
    dataset = create_multimodal_dataset(script_args, training_args)
    
    # Split dataset if needed
    if hasattr(dataset, 'train_test_split'):
        dataset_splits = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_splits['train']
        eval_dataset = dataset_splits['test'] if training_args.eval_strategy != "no" else None
    else:
        train_dataset = dataset
        eval_dataset = None

    # Initialize the OnlineDPOTrainer
    print("Initializing OnlineDPOTrainer...")
    trainer = OnlineDPOTrainer(
        model=model,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        reward_processing_class=reward_processor,
        peft_config=get_peft_config(model_args),
    )

    # Add completion logging callback for evaluation
    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens,
            do_sample=True,
            temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    # Start training
    print("Starting DPO online training...")
    trainer.train()

    # Save the trained model
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    # Push to hub if specified
    if training_args.push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()