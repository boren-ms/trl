"""
Example usage for dpo_online_bias.py

Basic Online DPO with Bias:
python trl/scripts/dpo_online_bias.py \
    --model_name_or_path microsoft/Phi-4-multimodal-instruct \
    --job_name phi4_dpo_bias \
    --train_data '{"jsonl_path": "train.jsonl", "bias_key": "biasing_words"}' \
    --eval_data '{"jsonl_path": "eval.jsonl", "bias_key": "biasing_words"}' \
    --output_dir ./outputs/phi4_dpo_bias \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --bf16 true
"""