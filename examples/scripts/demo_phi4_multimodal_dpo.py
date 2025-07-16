#!/usr/bin/env python3
"""
Demo script for DPO online training with Phi4-MultiModal model.

This script demonstrates how to use the DPO online training for phi4-multimodal
model with audio and text prompts. It includes mock data generation for testing
without requiring actual model weights or large datasets.

Usage:
    python examples/scripts/demo_phi4_multimodal_dpo.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add TRL to path
sys.path.insert(0, '/home/runner/work/trl/trl')

def create_mock_audio_file(duration=2.0, sample_rate=16000):
    """Create a mock audio file for testing."""
    import soundfile as sf
    
    # Generate simple sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sample_rate)
    temp_file.close()
    
    return temp_file.name

def create_mock_dataset(num_samples=5):
    """Create a mock dataset for demonstration."""
    dataset_samples = []
    audio_files = []
    
    print("Creating mock audio files...")
    for i in range(num_samples):
        # Create mock audio file
        audio_file = create_mock_audio_file()
        audio_files.append(audio_file)
        
        # Create sample data structure
        sample = {
            "prompt": f"<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>",
            "audio_path": audio_file,
            "text": f"This is sample transcription number {i+1}.",
            "id": f"sample_{i:03d}",
            "chosen": [{"role": "assistant", "content": f"This is sample transcription number {i+1}."}],
            "rejected": [{"role": "assistant", "content": f"This is wrong transcription number {i+1}."}]
        }
        dataset_samples.append(sample)
    
    return dataset_samples, audio_files

def demonstrate_configuration():
    """Demonstrate configuration file usage."""
    print("\n" + "="*60)
    print("CONFIGURATION DEMONSTRATION")
    print("="*60)
    
    # Show the configuration file
    config_path = "/home/runner/work/trl/trl/examples/configs/audio_dataset_config.json"
    if os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Configuration contents:")
        print(json.dumps(config, indent=2))
    else:
        print("Configuration file not found!")

def demonstrate_command_line():
    """Demonstrate command line usage."""
    print("\n" + "="*60)
    print("COMMAND LINE DEMONSTRATION")
    print("="*60)
    
    # Show example commands
    examples = [
        {
            "name": "Basic OpenASR training",
            "command": """python examples/scripts/dpo_online_phi4_multimodal.py \\
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \\
    --dataset_name openasr \\
    --learning_rate 5.0e-7 \\
    --output_dir phi4-multimodal-online-dpo \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 8 \\
    --warmup_ratio 0.1 \\
    --max_new_tokens 128 \\
    --use_peft \\
    --lora_target_modules=all-linear"""
        },
        {
            "name": "Custom dataset configuration",
            "command": """python examples/scripts/dpo_online_phi4_multimodal.py \\
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \\
    --dataset_name openasr \\
    --dataset_config_path examples/configs/audio_dataset_config.json \\
    --learning_rate 5.0e-7 \\
    --output_dir phi4-multimodal-online-dpo \\
    --use_peft"""
        },
        {
            "name": "Memory-optimized training",
            "command": """python examples/scripts/dpo_online_phi4_multimodal.py \\
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \\
    --dataset_name openasr \\
    --use_peft \\
    --load_in_4bit \\
    --gradient_checkpointing \\
    --per_device_train_batch_size 1 \\
    --gradient_accumulation_steps 32"""
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(example['command'])

def demonstrate_mock_training():
    """Demonstrate the training process with mock data."""
    print("\n" + "="*60)
    print("MOCK TRAINING DEMONSTRATION")
    print("="*60)
    
    try:
        # Create mock dataset
        print("Step 1: Creating mock dataset...")
        mock_samples, audio_files = create_mock_dataset(num_samples=3)
        print(f"Created {len(mock_samples)} mock samples")
        
        # Show sample structure
        print("\nSample data structure:")
        sample = mock_samples[0]
        print(f"  - Prompt: {sample['prompt'][:50]}...")
        print(f"  - Audio path: {sample['audio_path']}")
        print(f"  - Text: {sample['text']}")
        print(f"  - ID: {sample['id']}")
        
        # Import the script modules
        print("\nStep 2: Importing training modules...")
        from examples.scripts.dpo_online_phi4_multimodal import (
            MultimodalScriptArguments, 
            create_multimodal_dataset
        )
        print("✓ Successfully imported training modules")
        
        # Mock the training components
        print("\nStep 3: Setting up mock training components...")
        
        # Mock script arguments
        mock_script_args = Mock(spec=MultimodalScriptArguments)
        mock_script_args.dataset_name = "openasr"
        mock_script_args.dataset_config_path = None
        mock_script_args.max_train_samples = 10
        mock_script_args.dataset_streaming = False
        
        mock_training_args = Mock()
        
        # Mock dataset creation
        with patch('examples.scripts.dpo_online_phi4_multimodal.create_audio_dataset') as mock_create:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=len(mock_samples))
            mock_dataset.__getitem__ = Mock(side_effect=lambda i: mock_samples[i])
            mock_create.return_value = mock_dataset
            
            result_dataset = create_multimodal_dataset(mock_script_args, mock_training_args)
            print("✓ Successfully created mock dataset")
            print(f"  - Dataset length: {len(result_dataset)}")
        
        print("\nStep 4: Training simulation complete!")
        print("In a real scenario, this would:")
        print("  - Load the actual Phi4-MultiModal model")
        print("  - Process audio files and extract features")
        print("  - Generate completions using the model")
        print("  - Apply DPO loss to optimize preferences")
        print("  - Save the fine-tuned model")
        
    except Exception as e:
        print(f"Error in mock training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary audio files
        print("\nCleaning up temporary files...")
        for audio_file in audio_files:
            try:
                os.unlink(audio_file)
            except:
                pass

def demonstrate_features():
    """Demonstrate key features of the implementation."""
    print("\n" + "="*60)
    print("KEY FEATURES DEMONSTRATION")
    print("="*60)
    
    features = [
        {
            "name": "Multimodal Support",
            "description": "Handles both audio and text inputs seamlessly using Phi4-MultiModal model"
        },
        {
            "name": "Online DPO",
            "description": "Generates completions on-the-fly and learns from preferences in real-time"
        },
        {
            "name": "Flexible Dataset Loading",
            "description": "Supports OpenASR datasets (LibriSpeech) and custom TSV formats"
        },
        {
            "name": "LoRA/PEFT Integration",
            "description": "Memory-efficient training with parameter-efficient fine-tuning"
        },
        {
            "name": "Reward Model Support",
            "description": "Can use external reward models for preference evaluation"
        },
        {
            "name": "Bias Simulation",
            "description": "Built-in preference simulation through error injection for training data"
        },
        {
            "name": "Memory Optimization",
            "description": "Supports 4-bit quantization, gradient checkpointing, and other optimizations"
        }
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"\n{i}. {feature['name']}")
        print(f"   {feature['description']}")

def main():
    """Run the demonstration."""
    print("🎯 DPO Online Training for Phi4-MultiModal - DEMONSTRATION")
    print("="*80)
    print("This demo shows how to use online DPO training with audio and text prompts")
    print("for Microsoft's Phi4-MultiModal model.")
    
    try:
        # Run demonstrations
        demonstrate_features()
        demonstrate_configuration() 
        demonstrate_command_line()
        demonstrate_mock_training()
        
        print("\n" + "="*80)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Install required dependencies: pip install blobfile soundfile more-itertools shortuuid wandb")
        print("2. Prepare your audio dataset in OpenASR or TSV format")
        print("3. Run the training script with your desired configuration")
        print("4. Monitor training progress and adjust hyperparameters as needed")
        print("\nFor more details, see: examples/scripts/README_phi4_multimodal_dpo.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())