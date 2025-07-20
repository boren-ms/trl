"""
Example configuration and usage for dpo_online_bias.py

This demonstrates how to use the new Online DPO training script with bias support for phi4-MM model.
"""

# Example command line usage:
"""
python trl/scripts/dpo_online_bias.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --job_name phi4_dpo_bias_training \
    --project biasing_experiments \
    --lora_merged true \
    --output_dir ./outputs/phi4_dpo_bias \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --eval_strategy steps \
    --max_new_tokens 64 \
    --temperature 0.9 \
    --bf16 true
"""

# Example configuration for training data:
train_data_config = {
    "jsonl_path": "path/to/your/bias_training_data.jsonl",
    "bias_key": "biasing_words",
    "tag": "*",
    "data_dir": "path/to/audio/files",
    "streaming": False,
    "num_egs": 1000,
}

# Example configuration for evaluation data:
eval_data_config = {
    "jsonl_path": "path/to/your/bias_eval_data.jsonl", 
    "bias_key": "biasing_words",
    "tag": "*",
    "data_dir": "path/to/audio/files",
    "streaming": False,
    "num_egs": 100,
}

# Key features of the implementation:
"""
1. Supports phi4-MM multimodal model with processor instead of just tokenizer
2. Handles LoRA adapters for speech and vision modalities
3. Integrates with bias-aware audio datasets and metrics
4. Supports wandb logging with proper experiment tracking
5. Compatible with Online DPO training workflow
6. Includes proper error handling and fallbacks
"""

# Differences from regular dpo_online.py:
"""
1. Uses AutoProcessor for multimodal support instead of just AutoTokenizer
2. Includes LoRA adapter management for phi4-MM model
3. Integrates audio dataset creation and bias metrics evaluation
4. Includes comprehensive wandb experiment tracking with resume capability
5. Designed specifically for bias-aware training workflows
"""