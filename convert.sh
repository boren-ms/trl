## Inference code
1. Merge checkpoint and convert to HF format
bash
# Merge post-training model to PyTorch version
/home/weijianxu/code/Docs/phi-o/vision-speech/inference/hf_inference/official_final_trial2/verify_and_merge.ipynb

Output: /home/weijianxu/data/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore



# Prepare config from the vision-speech post-training
cp /home/weijianxu/data/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.2abl1-2.1k-speech-shadow50k-postsr002-posttrain-vision12k-trial2/config.json /home/weijianxu/data/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore

AZCOPY_AUTO_LOGIN_TYPE=AZCLI azcopy sync /home/weijianxu/data/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore/ https://tsstd01uks.blob.core.windows.net/data/users/weijianxu/phi-o/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore/ --recursive

# Convert PyTorch model to HF model

# Eval - debug
# Note: Use MoE weijian/dev as I have already migrated the two lora impl
cd /home/weijianxu/code/phi-o/MoE

# python -m debugpy --listen 0.0.0.0:9310 --wait-for-client \
export MODEL_PATH="/mnt/weijian_mount/tsstd01uks_data/users/weijianxu/phi-o/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore" && \
export OUTPUT_ROOT="/home/weijianxu/code/phi-o/MoE/output/phyagi_results" && \
VSDB_JSONL_PATH="/mnt/weijian_mount/phidataweub_vqa-tts/llava/llava-tts-rand-voice-single_image_single_audio-training_data_in_eval_format/llava-tts-train-subset.jsonl" \
VSDB_PROMPT=$'<|image_1|><|audio_1|>' \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="/home/weijianxu/code/phi-o/MoE:/home/weijianxu/code/phi-o/MoE/LMMBenchmarkAPI" \
PORT=55550 \
python \
    scripts/eval/eval_tasks_api.py \
    /home/weijianxu/code/phi-o/Configs/eval_configs/new_200k_vocab/lmm_bench-hd-128k-phi3-new200kvocab-longrope-vision_speech+vision_only_inplace_siglip_dyhd36_hf_convert.yaml \
    $MODEL_PATH \
    --agg \
    --data-root /mnt/weijian_mount/macamltwu3_instruct-tuning/eval_api \
    --output_dir ${OUTPUT_ROOT}/debug \
    --convert-hf 

Missing keys: []     -> Need to ensure both missing keys and unexpected keys are empty
Unexpected keys: []

Output: /mnt/weijian_mount/tsstd01uks_data/users/weijianxu/phi-o/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore/huggingface


2. Inference
bash
mv /home/weijianxu/data/MoE_weijian_phio-final-trial2-hf /home/weijianxu/data/MoE_weijian_phio-final_2-newtxtsftmore-hf
# create a new branch

rm /home/weijianxu/data/MoE_weijian_phio-final_2-newtxtsftmore-hf/hf-models/phio/*.safetensors
rm /home/weijianxu/data/MoE_weijian_phio-final_2-newtxtsftmore-hf/hf-models/phio/model.safetensors.index.json   

# copy tensors and index file
cp /mnt/weijian_mount/tsstd01uks_data/users/weijianxu/phi-o/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore/huggingface/hf/*.safetensors /home/weijianxu/data/MoE_weijian_phio-final_2-newtxtsftmore-hf/hf-models/phio/

cp /mnt/weijian_mount/tsstd01uks_data/users/weijianxu/phi-o/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore/huggingface/hf/model.safetensors.index.json /home/weijianxu/data/MoE_weijian_phio-final_2-newtxtsftmore-hf/hf-models/phio/

# manually update the config.json based on /home/weijianxu/data/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-trial2/config.json

# transformers >= 4.46.1
# pip install transformers==4.47.0 backoff peft==0.13.2
# pip3 install torch torchvision torchaudio
# MAX_JOBS=12 pip install flash-attn --no-build-isolation

cd /home/weijianxu/data/MoE_weijian_phio-longrope-ablation_2-hf/hf-models/phio/
CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 0.0.0.0:9310 --wait-for-client \
    sample_inference_phio.py

CUDA_VISIBLE_DEVICES=1 python sample_inference_phio.py

# Ref: https://git-lfs.com/
git lfs install 
git lfs track "*.safetensors"
# git add .gitattributes

AZCOPY_AUTO_LOGIN_TYPE=AZCLI \
AZCOPY_CONCURRENCY_VALUE=50000 azcopy sync "/home/weijianxu/data/MoE_weijian_phio-final_2-newtxtsftmore-hf/hf-models/phio/" "https://llmpretrainingwus3.blob.core.windows.net/users/weijianxu/phi-o/vision-speech-merged-pretraining/official_run/Phio-SFT-long-001-DPO-002/merged-vision-mframerc1.4abl1-2.2k-speech-shadow50k-postsr002-posttrain-vision12k-newtxtsftmore/vllm_lora/MoE_weijian_phio-final_2-newtxtsftmore-hf/hf-models/phio/" --recursive  --check-md5
