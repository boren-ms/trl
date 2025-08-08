#!/bin/bash

# This script is used to fetch results from a specific run of the GRPO trainer.
# python ./wandb_result.py --metric eval search grpo_ls_m1000_p9_n10_seed0_e2
# python ./wandb_result.py --metric metric search grpo_ls_m1000_p9_n10_seed0_e2
# python ./wandb_result.py --metric metric search grpo_ls_m1000_p9_n10_seed_e2_t12
# python ./wandb_result.py --metric metric_vllm search grpo_ls_m1000_p9_n10_zero_e2_t12
# python ./wandb_result.py --metric metric_vllm search grpo_ls_m1000_p9_n10_seed_e2_t12_simple_err_lora
# python ./wandb_result.py --metric metric_vllm search 5000_hf
# python ./wandb_result.py --metric metric_vllm search grpo_ls_m1000_p9_n10_seed0_e2_t12_simple_err_lora_G2x8
# python ./wandb_result.py --metric metric search grpo_ls_m1000_p9_n10_seed0_e2_t12_lower_err
# python ./wandb_result.py --metric metric_vllm search grpo_ls_m1000_p9_n10_zero_e2_t12_notag
# python ./wandb_result.py --metric metric_vllm search Phi-4-multimodal-instruct
# python ./wandb_result.py --metric metric_vllm search grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n10_sc3k_ctx8_G4x8
# python ./wandb_result.py --metric metric_vllm search grpo_ls_m1000_p9_n10_zero_e2_t12_notag_simple_err3_lora_G2x8
# python ./wandb_result.py --metric metric_vllm search Phi-4-multimodal-instruct
# python ./wandb_result.py --metric metric_vllm search Phi4-7b-ASR-2506-v2
# python ./wandb_result.py --metric eval search 'grpo_ls_m1000_seed_e3_simple_err_ga8_t15_n10x2_sc3k_ctx9_G1x8|grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n10_sc3k_ctx8_G4x8'
python ./wandb_result.py --metric metric_vllm search 'grpo_ls_m1000_p9_n10_zero_e2_t12_notag|grpo_ls_m1000_zero_e1_simple_err_ga8_t12_n10_sc3k_ctx8_G4x8'
