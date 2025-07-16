# DPO Online Training for Phi4-MultiModal with Audio and Text

This directory contains the implementation for online Direct Preference Optimization (DPO) training specifically designed for Microsoft's Phi4-MultiModal model with support for both audio and text prompts.

## Overview

The `dpo_online_phi4_multimodal.py` script enables online DPO training for multimodal scenarios where the model processes both audio input and text prompts. This is particularly useful for tasks like:

- Audio transcription with preference optimization
- Speech-to-text with quality preferences
- Audio understanding with human feedback
- Multimodal preference learning

## Features

- **Multimodal Support**: Handles both audio and text inputs seamlessly
- **Online DPO**: Generates completions on-the-fly and learns from preferences
- **Flexible Dataset Loading**: Supports OpenASR datasets and custom TSV formats
- **LoRA/PEFT Integration**: Memory-efficient training with parameter-efficient fine-tuning
- **Reward Model Support**: Can use external reward models for preference evaluation
- **Bias Simulation**: Built-in preference simulation for training data generation

## Quick Start

### Basic Usage

```bash
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
```

### With Custom Dataset Configuration

```bash
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --dataset_name openasr \
    --dataset_config_path examples/configs/audio_dataset_config.json \
    --learning_rate 5.0e-7 \
    --output_dir phi4-multimodal-online-dpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --use_peft
```

### With TSV Dataset

```bash
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --dataset_name tsv \
    --tsv_paths /path/to/your/audio_data.tsv \
    --learning_rate 5.0e-7 \
    --output_dir phi4-multimodal-online-dpo \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --use_peft
```

## Configuration

### Dataset Configuration

The script supports loading dataset configurations from JSON files. See `examples/configs/audio_dataset_config.json` for an example:

```json
{
  "dataset_name": "openasr",
  "name": "librispeech",
  "split": "test.clean",
  "num_egs": 1000,
  "streaming": false,
  "biasing": {
    "bias_prob": 0.9,
    "hit_prob": 0.9,
    "max_piece_len": 1,
    "max_num": 2
  },
  "simu_perference": {
    "error_range": [0.1, 0.25],
    "delete_prob": 0.1,
    "substitute_prob": 0.1,
    "insert_prob": 0.05
  },
  "load_audio": true
}
```

### Key Parameters

- `--dataset_name`: Choose between `openasr` (for LibriSpeech) or `tsv` (for custom data)
- `--dataset_config_path`: Path to JSON config file for advanced dataset configuration
- `--tsv_paths`: List of TSV file paths when using custom TSV datasets
- `--max_train_samples`: Limit the number of training samples
- `--use_peft`: Enable LoRA/PEFT for memory-efficient training
- `--reward_model_path`: Path to reward model for preference evaluation
- `--judge`: Use external judges like "openai" or "hf" for preference evaluation

## Dataset Formats

### OpenASR Format

The OpenASR format uses the ESB (End-to-end Speech Benchmark) datasets, particularly LibriSpeech. The script automatically handles:
- Audio loading and preprocessing
- Text transcription targets
- Bias word injection for preference generation
- Error simulation for negative examples

### TSV Format

For custom datasets, use TSV files with the following structure:
- Column 1: ID
- Column 2: Audio file paths (as Python list string)
- Column 3: Messages (as Python list of dictionaries)

Example TSV row:
```
sample_001	["/path/to/audio.wav"]	[{"messages": [{"role": "user", "content": "Transcribe this audio"}, {"role": "assistant", "content": "Hello world"}]}]
```

## Requirements

The script requires additional dependencies beyond the base TRL installation:

```bash
pip install blobfile soundfile more-itertools shortuuid wandb
```

## Advanced Usage

### With Reward Model

```bash
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --reward_model_path path/to/your/reward/model \
    --dataset_name openasr \
    --learning_rate 5.0e-7 \
    --output_dir phi4-multimodal-online-dpo-with-rm \
    --use_peft
```

### With External Judge

```bash
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --judge openai \
    --dataset_name openasr \
    --learning_rate 5.0e-7 \
    --output_dir phi4-multimodal-online-dpo-openai \
    --use_peft
```

### Memory Optimization

For large models or limited VRAM:

```bash
python examples/scripts/dpo_online_phi4_multimodal.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --dataset_name openasr \
    --use_peft \
    --load_in_4bit \
    --gradient_checkpointing \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --output_dir phi4-multimodal-online-dpo-4bit
```

## Architecture

The script combines several key components:

1. **Phi4-MultiModal Model**: Handles both speech and vision inputs
2. **Audio Dataset Processing**: Loads and preprocesses audio data with text annotations
3. **Online DPO Trainer**: Generates completions and optimizes preferences in real-time
4. **Preference Simulation**: Creates chosen/rejected pairs through error injection
5. **LoRA Adapters**: Efficient training with parameter-efficient fine-tuning

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use `--load_in_4bit`, increase `--gradient_accumulation_steps`, decrease `--per_device_train_batch_size`
2. **Dataset Loading**: Ensure audio files are accessible and in supported formats
3. **Model Loading**: Verify you have access to the Phi4-MultiModal model weights

### Performance Tips

- Use `--gradient_checkpointing` for memory efficiency
- Enable LoRA with `--use_peft` for faster training
- Adjust `--max_new_tokens` based on your task requirements
- Use `--bf16` for better performance on supported hardware

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{trl-phi4-multimodal-dpo,
  title={DPO Online Training for Phi4-MultiModal with Audio and Text},
  author={TRL Team},
  year={2025},
  publisher={Hugging Face},
  url={https://github.com/huggingface/trl}
}
```