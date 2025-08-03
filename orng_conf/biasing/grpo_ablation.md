# Ablation Study for GRPO

This document outlines the ablation experiments conducted to analyze different components of the GRPO (Group Relative Policy Optimization) approach.

## Temperature Analysis

Investigating the impact of different temperature settings on model performance:

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t10_simple_err_lora_ga8.yaml` (temp=1.0)
2. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t12_simple_err_lora_ga8.yaml` (temp=1.2)
3. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t15_simple_err_lora_ga8.yaml` (temp=1.5)
4. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t20_simple_err_lora_ga8.yaml` (temp=2.0)

## Number of Completions

Evaluating how the number of generated completions affects training:

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n5_seed_e2_t12_simple_err_lora_ga8.yaml` (n=5 completions)
2. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t12_simple_err_lora_ga8.yaml` (n=10 completions)
3. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n20_seed_e2_t12_simple_err_lora_ga8.yaml` (n=20 completions)
4. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n30_seed_e2_t12_simple_err_lora_ga8.yaml` (n=30 completions) failed due to OOM

## Reward Function Comparison

Testing different reward function formulations:

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_err_lora_ga8.yaml` (word+bias error reward)
2. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_bias_err_lora_ga8.yaml` (bias error reward) #worse as expected
3. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_word_err_lora_ga8.yaml` (word error reward)

## Different seed model

Testing the impact of using a different seed model for training: 
Seed models include: zero, seed0, seed.

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_err_lora_ga8.yaml` (match sft+RL)
2. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_zero_e2_t12_notag.yaml` (zero RL)
3. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed0_e2_t12_lower_err.yaml` (mismatch SFT + match RL)

## high entropy only 

Play the paper https://arxiv.org/abs/2506.01939
Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

1. ✓ `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e2_t12_simple_err_lora_ga8.yaml` (80/20)
2. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n10_seed_e1_t12_simple_err_lora_ga8_entropy80.yaml` (high entropy only) 80/20
3. Running: `orng_conf/biasing/grpo_ls_m1000_p9_n20_seed_e1_t20_simple_err_lora_ga8_entropy80.yaml` (entropy=80, t=1.5, n=20)

## Rare word improvement

design a reward function to concentrate on rare words, but without additional biasing list as input.

1. TODO: implement the dataset to tag rare words for each utterance.

## add noisy completions
1. add bad completions to replace repeated completions. [Do not work]


## Evaluation on out-of-domain dataset

1. evaluation on out-of-domain dataset (internal entity dataset)
   1. grpo_ls_m1000_p9_n10_zero_e2_notag_err_lora [zeroRL]
   2. grpo_ls_m1000_p9_n10_seed_e2_t12_simple_err_lora [seed RL]
