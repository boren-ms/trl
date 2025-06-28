#!/bin/bash

set -x 
# https://tsstd01uks.blob.core.windows.net/data/users/ruchaofan/Phi-omni-checkpoints/Phi-4-7b-ASR-merged/

# from vadim
# vllm serve /home/azureuser/cloudfiles/code/Users/vadimma/models/Phi-4-7b-ASR-merged/ --host 127.0.0.1 --port 26500 --tensor-parallel-size 1 --trust-remote-code --load-format auto --max-model-len 8192 --limit-mm-per-prompt audio=10


# model_path=/home/boren/data/hf_models/Phi-4-7b-ASR-merged
# model_path=/home/boren/data/hf_models/Phi-4-multimodal-instruct
# model_path=/home/boren/data/hf_models/phi4_mm_bias_merged
# use sampling/merge_hf_lora.py to merge lora weights into the model.
model_path=/home/boren/data/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf_merged
vllm serve ${model_path} --host 127.0.0.1 --port 26500 --tensor-parallel-size 1 --trust-remote-code --load-format auto --max-model-len 8192 --limit-mm-per-prompt audio=10

# with lora
# model_path=/home/boren/data/hf_models/Phi-4-multimodal-instruct
# vllm serve ${model_path} --host 127.0.0.1 --port 26500 --tensor-parallel-size 1 --trust-remote-code --load-format auto --max-model-len 32000 \
#      --enable-lora --max-lora-rank 512 --lora-extra-vocab-size 256 --limit-mm-per-prompt audio=10 --max-loras 5 --lora-modules "speech=${model_path}/speech-lora"