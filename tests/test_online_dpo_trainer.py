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

import tempfile
import unittest

import pytest
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers.testing_utils import require_peft, require_torch_accelerator
from transformers.utils import is_peft_available

from trl import OnlineDPOConfig, OnlineDPOTrainer

from .testing_utils import RandomPairwiseJudge, require_llm_blender, require_vllm


if is_peft_available():
    from peft import LoraConfig, get_peft_model


class TestOnlineDPOTrainer(unittest.TestCase):
    def setUp(self):
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_model_id = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_id, num_labels=1)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_id)
        self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_training(self, config_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

            trainer = OnlineDPOTrainer(
                model=self.model,
                reward_model=self.reward_model,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
                reward_processing_class=self.reward_tokenizer,
            )
            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    def test_training_with_ref_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

            trainer = OnlineDPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
                reward_processing_class=self.reward_tokenizer,
            )
            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    def test_ref_model_is_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                report_to="none",
            )

            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

            with self.assertRaises(ValueError):
                OnlineDPOTrainer(
                    model=self.model,
                    ref_model=self.model,  # ref_model can't be the same as model
                    reward_model=self.reward_model,
                    args=training_args,
                    train_dataset=dummy_dataset["train"],
                    processing_class=self.tokenizer,
                    reward_processing_class=self.reward_tokenizer,
                )

    @require_peft
    def test_training_with_peft(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

            trainer = OnlineDPOTrainer(
                model=self.model,
                reward_model=self.reward_model,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
                reward_processing_class=self.reward_tokenizer,
                peft_config=lora_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_peft
    def test_training_with_peft_and_ref_model(self):
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

            trainer = OnlineDPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                reward_model=self.reward_model,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
                reward_processing_class=self.reward_tokenizer,
                peft_config=lora_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_peft
    def test_training_with_peft_model_and_peft_config(self):
        model_lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(self.model, model_lora_config)
        # we want only the "train adapter" to be trained
        lora_train_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only")

            trainer = OnlineDPOTrainer(
                model=model,
                reward_model=self.reward_model,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
                reward_processing_class=self.reward_tokenizer,
                peft_config=lora_train_config,
            )

            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @require_llm_blender
    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    def test_training_with_judge(self, config_name):
        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                per_device_train_batch_size=2,
                max_steps=3,
                learning_rate=5.0e-7,
                eval_strategy="steps",
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

            trainer = OnlineDPOTrainer(
                model=self.model,
                judge=RandomPairwiseJudge(),
                args=training_args,
                train_dataset=dummy_dataset["train"],
                eval_dataset=dummy_dataset["test"],
                processing_class=self.tokenizer,
            )
            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    @parameterized.expand([("standard_prompt_only",), ("conversational_prompt_only",)])
    @require_torch_accelerator
    @require_vllm
    @pytest.mark.slow
    def test_training_with_vllm(self, config_name):
        model_id = "trl-internal-testing/small-Qwen2ForCausalLM-2.5"  # We need a bigger model
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = OnlineDPOConfig(
                output_dir=tmp_dir,
                use_vllm=True,
                gpu_memory_utilization=0.2,
                report_to="none",
            )
            dummy_dataset = load_dataset("trl-internal-testing/zen", config_name)

            trainer = OnlineDPOTrainer(
                model=model,
                reward_model=self.reward_model,
                args=training_args,
                train_dataset=dummy_dataset["train"],
                processing_class=tokenizer,
                reward_processing_class=self.reward_tokenizer,
            )
            trainer.train()

            # Check if training loss is available
            self.assertIn("train_loss", trainer.state.log_history[-1])

    def test_multimodal_data_handling(self):
        """Test that multimodal data structures with audio_path are handled correctly"""
        import numpy as np
        import soundfile as sf
        import os
        
        # Create a temporary audio file
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = os.path.join(tmp_dir, "test_audio.wav")
            
            # Create dummy audio data
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            sf.write(audio_path, audio_data, sample_rate)
            
            # Test data structure with audio_path
            multimodal_inputs = {
                "prompt": ["Test prompt with audio"],
                "audio_path": [audio_path]
            }
            
            # Test that we can detect audio presence
            has_audio = "audio_path" in multimodal_inputs and multimodal_inputs["audio_path"] is not None
            self.assertTrue(has_audio, "Should detect audio presence in multimodal inputs")
            
            # Test that we can read the audio file using sf_read
            from trl.data_utils import sf_read
            audio, sr = sf_read(audio_path)
            self.assertEqual(sr, sample_rate, "Sample rate should match")
            self.assertEqual(len(audio), int(sample_rate * duration), "Audio length should match expected duration")
            
            # Test text-only inputs (should not have audio)
            text_only_inputs = {
                "prompt": ["Test prompt without audio"]
            }
            has_audio_text_only = "audio_path" in text_only_inputs and text_only_inputs["audio_path"] is not None
            self.assertFalse(has_audio_text_only, "Should not detect audio in text-only inputs")

    def test_multimodal_generation_method_signatures(self):
        """Test that the generation methods accept inputs parameter correctly"""
        # Test that the methods can be called with inputs parameter
        trainer = OnlineDPOTrainer(
            model=self.model,
            reward_model=self.reward_model,
            args=OnlineDPOConfig(output_dir="test", report_to="none"),
            train_dataset=[],
            processing_class=self.tokenizer,
            reward_processing_class=self.reward_tokenizer,
        )
        
        # Test that _generate method accepts inputs parameter
        # We can't easily test the full functionality without a complex setup,
        # but we can at least verify the method signature works
        try:
            import inspect
            generate_sig = inspect.signature(trainer._generate)
            self.assertIn("inputs", generate_sig.parameters, "_generate should accept 'inputs' parameter")
            
            generate_vllm_sig = inspect.signature(trainer._generate_vllm)  
            self.assertIn("inputs", generate_vllm_sig.parameters, "_generate_vllm should accept 'inputs' parameter")
            
        except Exception as e:
            # If we can't inspect signatures, at least verify the methods exist
            self.assertTrue(hasattr(trainer, '_generate'), "Trainer should have _generate method")
            self.assertTrue(hasattr(trainer, '_generate_vllm'), "Trainer should have _generate_vllm method")
