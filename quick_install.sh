pip uninstall -y torch torchvision torchaudio transformers flash-attn vllm trl
pip install  torch==2.6.0 torchvision torchaudio transformers==4.51.3 vllm trl peft tensorboardX blobfile soundfile
pip install  torch==2.6.0 flash-attn 
pip uninstall -y trl
pip install -e . --no-deps