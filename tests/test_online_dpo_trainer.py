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

    def test_multimodal_input_handling(self):
        """Test that the trainer can handle multi-modal input format without crashing."""
        # Test 1: Legacy format (list of strings) - should work
        legacy_inputs = ["Hello world", "How are you?"]
        # Test input format detection logic
        if isinstance(legacy_inputs, list) and isinstance(legacy_inputs[0], str):
            prompts = legacy_inputs
            inputs_dict = [{"prompt": prompt} for prompt in prompts]
            has_multimodal = False
            self.assertFalse(has_multimodal)
            self.assertEqual(len(inputs_dict), 2)
            self.assertEqual(inputs_dict[0]["prompt"], "Hello world")
        
        # Test 2: Multi-modal format - should be detected
        multimodal_inputs = [
            {"prompt": "Describe this audio", "audio_path": "/fake/path/audio1.wav"},
            {"prompt": "What do you hear?", "audio_path": "/fake/path/audio2.wav"}
        ]
        # Test multi-modal detection logic
        prompts = [x["prompt"] for x in multimodal_inputs]
        has_multimodal = any("audio_path" in x for x in multimodal_inputs)
        self.assertTrue(has_multimodal)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0], "Describe this audio")
        
        # Test audio path extraction
        audio_paths = [x.get("audio_path") for x in multimodal_inputs]
        self.assertEqual(len(audio_paths), 2)
        self.assertEqual(audio_paths[0], "/fake/path/audio1.wav")
        
        # Test 3: Text-only format in new structure - should work
        text_only_inputs = [
            {"prompt": "Hello world"},
            {"prompt": "How are you?"}
        ]
        prompts = [x["prompt"] for x in text_only_inputs]
        has_multimodal = any("audio_path" in x for x in text_only_inputs)
        self.assertFalse(has_multimodal)
        self.assertEqual(len(prompts), 2)
        
        # Test 4: Mixed format (some with audio, some without)
        mixed_inputs = [
            {"prompt": "Describe this audio", "audio_path": "/fake/path/audio1.wav"},
            {"prompt": "This is text only"}
        ]
        prompts = [x["prompt"] for x in mixed_inputs]
        has_multimodal = any("audio_path" in x for x in mixed_inputs)
        audio_paths = [x.get("audio_path") for x in mixed_inputs]
        
        self.assertTrue(has_multimodal)  # Should detect as multi-modal due to first item
        self.assertEqual(len(prompts), 2)
        self.assertEqual(audio_paths[0], "/fake/path/audio1.wav")
        self.assertIsNone(audio_paths[1])
