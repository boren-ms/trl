# Phi4 Multi-Modality Training Scripts

This directory contains specialized training scripts for phi4 multi-modality (vision-language) models that support both text and image inputs.

## Available Scripts

### 1. `online_dpo_vlm_phi4.py`
Online DPO training for phi4 vision-language models.

**Key Features:**
- Supports phi4 multi-modal architecture
- Online DPO training with real-time reward feedback
- Compatible with vision-language datasets
- Supports LoRA fine-tuning
- Uses AutoModelForVision2Seq and AutoProcessor

**Usage:**
```bash
python examples/scripts/online_dpo_vlm_phi4.py \
    --model_name_or_path microsoft/phi-4 \
    --reward_model_path your_reward_model \
    --dataset_name your_vlm_dataset \
    --learning_rate 5.0e-7 \
    --output_dir phi4-online-dpo-vlm \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --use_peft
```

### 2. `grpo_vlm_phi4.py`
GRPO training for phi4 vision-language models.

**Key Features:**
- Group Relative Policy Optimization for VLMs
- Support for custom reward functions
- Multi-modal input processing
- Configurable image and text column names

**Usage:**
```bash
python examples/scripts/grpo_vlm_phi4.py \
    --model_name_or_path microsoft/phi-4 \
    --reward_model_name_or_path your_reward_model \
    --dataset_name your_vlm_dataset \
    --learning_rate 5.0e-7 \
    --output_dir phi4-grpo-vlm \
    --per_device_train_batch_size 2 \
    --use_peft
```

### 3. `grpo_bias.py`
GRPO training with bias mitigation for phi4 models.

**Key Features:**
- Bias-aware training specifically designed for fairness
- Same functionality as grpo_vlm_phi4.py but with focus on bias reduction
- Suitable for responsible AI applications

**Usage:**
```bash
python examples/scripts/grpo_bias.py \
    --model_name_or_path microsoft/phi-4 \
    --reward_model_name_or_path your_reward_model \
    --dataset_name your_vlm_dataset \
    --learning_rate 5.0e-7 \
    --output_dir phi4-grpo-bias \
    --per_device_train_batch_size 2 \
    --use_peft
```

## Common Parameters

All scripts support these common parameters:

- `--model_name_or_path`: Path to the phi4 model (e.g., "microsoft/phi-4")
- `--dataset_name`: Name of the vision-language dataset 
- `--output_dir`: Directory to save the trained model
- `--use_peft`: Enable LoRA fine-tuning
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Steps to accumulate gradients
- `--learning_rate`: Learning rate for training
- `--warmup_ratio`: Warmup ratio for learning rate scheduler

## Model Requirements

These scripts are designed for:
- Phi4 multi-modal models with vision capabilities
- Models compatible with AutoModelForVision2Seq
- Datasets containing both images and text

## Dependencies

The scripts require:
- transformers >= 4.53.0
- trl >= 0.20.0
- torch >= 2.0.0
- datasets >= 3.0.0