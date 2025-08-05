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
python ./wandb_result.py --metric metric_vllm search grpo_ls_m1000_p9_n10_zero_e2_t12_notag

