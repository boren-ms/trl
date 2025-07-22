#!/usr/bin/env python3
"""
Online DPO Training Workflow Example with Detailed Function Call Logging

This script demonstrates the complete Online DPO training workflow with comprehensive
logging of all function calls and their interactions. This serves as both a practical
example and a reference for understanding the training process.

Usage:
    python examples/scripts/online_dpo_workflow_example.py \
        --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
        --dataset_name trl-lib/ultrafeedback-prompt \
        --judge pair_rm \
        --output_dir ./online-dpo-workflow-example \
        --max_steps 10 \
        --logging_steps 1 \
        --per_device_train_batch_size 2

For a complete documentation of this workflow, see:
- docs/source/online_dpo_training_workflow.md
- docs/source/online_dpo_workflow_diagrams.md
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from trl import (
    OnlineDPOConfig,
    OnlineDPOTrainer,
    PairRMJudge,
    ModelConfig,
    ScriptArguments,
    TrlParser,
)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_function_call(func_name: str, args: Dict[str, Any] = None, timing: bool = True):
    """Decorator to log function calls with detailed information."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args_tuple, **kwargs):
            start_time = time.time() if timing else None
            
            # Log function entry
            logger.info(f"🔵 ENTERING: {func_name}")
            if args:
                logger.info(f"   📋 Arguments: {args}")
            
            # Execute function
            try:
                result = func(*args_tuple, **kwargs)
                
                # Log successful completion
                elapsed = f" ({time.time() - start_time:.3f}s)" if timing else ""
                logger.info(f"✅ COMPLETED: {func_name}{elapsed}")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ ERROR in {func_name}: {str(e)}")
                raise
                
        return wrapper
    return decorator


class DetailedOnlineDPOTrainer(OnlineDPOTrainer):
    """
    Extended OnlineDPOTrainer with comprehensive logging of all function calls.
    
    This class wraps key methods to provide detailed insights into the training workflow.
    """
    
    def __init__(self, *args, **kwargs):
        logger.info("🚀 INITIALIZING OnlineDPOTrainer")
        logger.info("   📝 Setting up models, tokenizers, and training components...")
        
        super().__init__(*args, **kwargs)
        
        logger.info("✅ OnlineDPOTrainer initialization complete")
        logger.info(f"   🎯 Policy model: {self.model.__class__.__name__}")
        logger.info(f"   🔄 Reference model: {type(self.ref_model).__name__ if self.ref_model else 'PEFT disabled adapter'}")
        logger.info(f"   🏆 Scoring method: {'Judge' if self.judge else 'Reward Model'}")
        logger.info(f"   ⚡ Generation method: {'vLLM' if self.args.use_vllm else 'Standard'}")
    
    @log_function_call("training_step", timing=True)
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Enhanced training step with detailed logging."""
        logger.info("📊 Starting training step")
        
        model.train()
        prompts = inputs["prompt"]
        batch_size = len(prompts)
        
        logger.info(f"   📝 Batch size: {batch_size}")
        logger.info(f"   📄 Sample prompt: {prompts[0][:100]}...")
        
        # Step 1: Generation Phase
        logger.info("🎭 PHASE 1: Generation")
        if self.args.use_vllm:
            logger.info("   ⚡ Using vLLM generation")
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm_logged(model, inputs)
        else:
            logger.info("   🔄 Using standard generation")
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_logged(model, inputs)
        
        # Log generation results
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        logger.info(f"   ✨ Generated {len(completions)} completions")
        logger.info(f"   📄 Sample completion: {completions[0][:100]}...")
        
        contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)
        eos_rate = contain_eos_token.float().mean().item()
        logger.info(f"   🔚 EOS token rate: {eos_rate:.2%}")
        
        # Step 2: Forward Pass Phase
        logger.info("🧠 PHASE 2: Forward Pass")
        logger.info("   🎯 Computing policy model logprobs")
        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        
        with torch.no_grad():
            logger.info("   🔄 Computing reference model logprobs")
            if self.ref_model is not None:
                ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:
                logger.info("     💡 Using PEFT: disabling adapter for reference")
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        
        # Step 3: Scoring Phase  
        logger.info("🏆 PHASE 3: Scoring")
        device = logprobs.device
        
        if self.judge is not None:
            logger.info("   👨‍⚖️ Using judge for scoring")
            ranks_of_first_completion = self._judge_scoring_logged(prompts, completions, batch_size)
            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
        else:
            logger.info("   🎖️ Using reward model for scoring")
            mask = self._reward_model_scoring_logged(prompts, completions, batch_size, contain_eos_token, device)
        
        preference_rate = mask.float().mean().item()
        logger.info(f"   📊 First completion preferred: {preference_rate:.2%}")
        
        # Step 4: Loss Computation Phase
        logger.info("💰 PHASE 4: Loss Computation")
        loss = self._compute_dpo_loss_logged(logprobs, ref_logprobs, mask, completion_mask, batch_size, device)
        
        # Step 5: Backward Pass
        logger.info("⬅️ PHASE 5: Backward Pass")
        logger.info("   🔄 Computing gradients")
        self.accelerator.backward(loss)
        
        logger.info(f"✅ Training step complete - Loss: {loss.item():.4f}")
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _generate_logged(self, model, inputs):
        """Standard generation with logging."""
        logger.info("   🔧 Processing inputs for generation")
        prompt_ids, prompt_mask, additional_inputs = self._process_inputs_for_generation(inputs)
        logger.info(f"     📏 Prompt length: {prompt_ids.size(1)}")
        
        logger.info("   🎭 Generating completions")
        with self.unwrap_model_for_generation(
            model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                **additional_inputs
            )
        
        completion_ids = output[:, prompt_ids.size(1):]
        completion_ids, completion_mask = self.truncate_right(
            completion_ids, self.processing_class.eos_token_id, self.processing_class.pad_token_id
        )
        
        logger.info(f"     📏 Average completion length: {completion_mask.sum(1).float().mean():.1f}")
        return prompt_ids, prompt_mask, completion_ids, completion_mask
    
    def _generate_vllm_logged(self, model, inputs):
        """vLLM generation with logging."""
        logger.info("   🔄 Loading latest weights into vLLM")
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(model.state_dict().items())
        
        logger.info("   🎭 Generating with vLLM")
        generation_prompts = self._prepare_prompts_for_generation(inputs)
        outputs = self.llm.generate(generation_prompts, self.generation_config, use_tqdm=False)
        
        logger.info(f"     📊 Generated {len(outputs)} outputs with 2 completions each")
        
        # Process outputs (implementation details logged)
        completion_ids = [list(output.outputs[i].token_ids) for i in range(2) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(2) for output in outputs]
        
        # Additional processing and tensor creation...
        # (Implementation continues as in original method)
        
    def _process_vllm_outputs_logged(self, prompt_ids, completion_ids):
        """Process vLLM outputs with logging."""
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id
        
        # Create mask and pad the prompt and completion
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        
        max_tokens = self.generation_config.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]
        
        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)
        
        logger.info(f"     📊 Processed tensors - Prompts: {prompt_ids.shape}, Completions: {completion_ids.shape}")
        
        return prompt_ids, prompt_mask, completion_ids, completion_mask
    
    def unwrap_model_for_generation(self, *args, **kwargs):
        """Wrapper for unwrap_model_for_generation context manager."""
        from trl.models.utils import unwrap_model_for_generation
        return unwrap_model_for_generation(*args, **kwargs)
    
    def truncate_right(self, *args, **kwargs):
        """Wrapper for truncate_right function."""
        from trl.trainer.utils import truncate_right
        return truncate_right(*args, **kwargs)
    
    def _judge_scoring_logged(self, prompts, completions, batch_size):
        """Judge-based scoring with logging."""
        from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
        import jinja2
        from trl.data_utils import is_conversational
        
        if is_conversational({"prompt": prompts[0]}):
            logger.info("     🗣️ Applying chat template for judge")
            environment = jinja2.Environment()
            template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
            prompts = [template.render(messages=prompt) for prompt in prompts]
            completions = [template.render(messages=completion) for completion in completions]
        
        logger.info("     ⚖️ Calling judge for ranking")
        ranks = self.judge.judge(
            prompts, list(zip(completions[:batch_size], completions[batch_size:]))
        )
        
        first_wins = sum(1 for rank in ranks if rank == 0)
        logger.info(f"     📊 Judge preferences: {first_wins}/{batch_size} first completions preferred")
        
        return ranks
    
    def _reward_model_scoring_logged(self, prompts, completions, batch_size, contain_eos_token, device):
        """Reward model scoring with logging."""
        from trl.data_utils import is_conversational, apply_chat_template
        from trl.trainer.utils import get_reward
        
        # Prepare inputs
        prompts = 2 * prompts  # Duplicate for both completions
        
        if is_conversational({"prompt": prompts[0]}):
            logger.info("     🗣️ Applying chat template for reward model")
            examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
            examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
            prompts = [example["prompt"] for example in examples]
            completions = [example["completion"] for example in examples]
        
        logger.info("     🔤 Tokenizing inputs for reward model")
        prompts_ids = self.reward_processing_class(
            prompts, padding=True, return_tensors="pt", padding_side="left"
        )["input_ids"].to(device)
        
        completions_ids = self.reward_processing_class(
            completions, padding=True, return_tensors="pt", padding_side="right"
        )["input_ids"].to(device)
        
        logger.info(f"     📏 Prompt length: {prompts_ids.size(1)}, Completion length: {completions_ids.size(1)}")
        
        # Score with reward model
        prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
        
        logger.info("     🎖️ Computing reward scores")
        with torch.inference_mode():
            _, scores, _ = get_reward(
                self.reward_model, prompt_completion_ids, 
                self.reward_processing_class.pad_token_id, prompts_ids.shape[1]
            )
            
            if self.args.missing_eos_penalty is not None:
                penalty_applied = (~contain_eos_token).sum().item()
                logger.info(f"     ⚠️ Applied EOS penalty to {penalty_applied} completions")
                scores[~contain_eos_token] -= self.args.missing_eos_penalty
        
        first_half, second_half = scores.split(batch_size)
        mask = first_half >= second_half
        
        logger.info(f"     📊 Average scores - First: {first_half.mean():.3f}, Second: {second_half.mean():.3f}")
        
        return mask
    
    def _compute_dpo_loss_logged(self, logprobs, ref_logprobs, mask, completion_mask, batch_size, device):
        """DPO loss computation with logging."""
        import torch.nn.functional as F
        
        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)
        
        logger.info("   📋 Organizing chosen/rejected completions")
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]
        
        # Mask padding and sum logprobs
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]
        
        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)
        
        logger.info("   🧮 Computing log ratios")
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum
        logits = pi_logratios - ref_logratios
        
        logger.info(f"     📊 Average logits: {logits.mean():.3f}")
        
        # Compute loss
        logger.info(f"   💰 Computing {self.args.loss_type} loss")
        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"Loss type {self.args.loss_type} not implemented")
        
        loss = losses.mean()
        logger.info(f"     💰 Loss: {loss.item():.4f}")
        
        # Log metrics
        self._log_training_metrics(logprobs, ref_logprobs, chosen_logprobs_sum, rejected_logprobs_sum, mask)
        
        return loss
    
    def _log_training_metrics(self, logprobs, ref_logprobs, chosen_logprobs_sum, rejected_logprobs_sum, mask):
        """Log detailed training metrics."""
        # KL divergence
        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        
        # Entropy
        mean_entropy = -logprobs.sum(1).mean()
        
        # Rewards
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_logprobs_sum.detach())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_logprobs_sum.detach())
        
        logger.info(f"   📈 Metrics - KL: {mean_kl:.4f}, Entropy: {mean_entropy:.4f}")
        logger.info(f"   🏆 Rewards - Chosen: {chosen_rewards.mean():.4f}, Rejected: {rejected_rewards.mean():.4f}")


def main():
    """Main training function with comprehensive workflow logging."""
    
    # Parse arguments
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    logger.info("🌟 STARTING Online DPO Training Workflow Example")
    logger.info("=" * 80)
    
    # Log configuration
    logger.info("📋 CONFIGURATION")
    logger.info(f"   🎯 Model: {model_args.model_name_or_path}")
    logger.info(f"   📚 Dataset: {script_args.dataset_name}")
    logger.info(f"   🏆 Judge: {training_args.judge}")
    logger.info(f"   📊 Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"   📈 Learning rate: {training_args.learning_rate}")
    logger.info(f"   🎭 Max new tokens: {training_args.max_new_tokens}")
    logger.info(f"   ⚡ Use vLLM: {training_args.use_vllm}")
    
    # Phase 1: Model and Tokenizer Setup
    logger.info("\n🔧 PHASE 1: Model and Tokenizer Setup")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        trust_remote_code=model_args.trust_remote_code,
    )
    logger.info(f"✅ Loaded policy model: {model.__class__.__name__}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"✅ Loaded tokenizer with vocab size: {len(tokenizer)}")
    
    # Phase 2: Judge Setup
    logger.info("\n🏆 PHASE 2: Judge Setup")
    judge = PairRMJudge()
    logger.info("✅ Initialized PairRM judge")
    
    # Phase 3: Dataset Loading
    logger.info("\n📚 PHASE 3: Dataset Loading")
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    logger.info(f"✅ Loaded dataset with {len(train_dataset)} training examples")
    
    # Phase 4: Trainer Initialization
    logger.info("\n🚀 PHASE 4: Trainer Initialization")
    trainer = DetailedOnlineDPOTrainer(
        model=model,
        judge=judge,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Phase 5: Training
    logger.info("\n🎯 PHASE 5: Training Loop")
    logger.info("=" * 80)
    
    trainer.train()
    
    # Phase 6: Model Saving
    logger.info("\n💾 PHASE 6: Model Saving")
    trainer.save_model(training_args.output_dir)
    logger.info(f"✅ Model saved to {training_args.output_dir}")
    
    logger.info("\n🎉 Training workflow completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()