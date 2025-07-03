pip uninstall -y torch torchvision torchaudio transformers flash-attn vllm trl
uv pip install --system  torch==2.6.0 torchvision torchaudio transformers==4.51.3 vllm trl peft tensorboardX blobfile soundfile
pip install   torch==2.6.0 flash-attn
# uv pip install --system  torch==2.6.0 flashinfer-python==0.2.6
pip uninstall -y trl
uv pip install --system  -e . --no-deps