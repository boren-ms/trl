# Ablation Study for GRPO

This document outlines the ablation experiments conducted to analyze different components of the GRPO (Group Relative Policy Optimization) approach.

## Temperature Analysis

Investigating the impact of different temperature settings on model performance:

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t10_simple_err_lora_ga8.yaml` (temp=1.0)
2. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t12_simple_err_lora_ga8.yaml` (temp=1.2)
3. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t15_simple_err_lora_ga8.yaml` (temp=1.5)

## Number of Completions

Evaluating how the number of generated completions affects training:

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n5_seed_e2_t12_simple_err_lora_ga8.yaml` (n=5 completions)
2. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t12_simple_err_lora_ga8.yaml` (n=10 completions)
3. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n20_seed_e2_t12_simple_err_lora_ga8.yaml` (n=20 completions)
4. TODO: `orng_conf/biasing/grpo_ls_m1000_p9_n30_seed_e2_t12_simple_err_lora_ga8.yaml` (n=30 completions) failed due to OOM

## Reward Function Comparison

Testing different reward function formulations:

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_err_lora_ga8.yaml` (word+bias error reward)
2. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_bias_err_lora_ga8.yaml` (bias error reward)
3. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_word_err_lora_ga8.yaml` (word error reward)