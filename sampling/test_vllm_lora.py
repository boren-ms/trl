# %%
import os
from typing import Any, NamedTuple, Optional
from vllm.lora.request import LoRARequest
import vllm
from pathlib import Path
from vllm import LLM, SamplingParams

# %%


# model_path = "/root/data/ckp/hf_models/phi4_mm_bias"
model_path = Path("/root/data/ckp/hf_models/Phi-4-multimodal-instruct")

# %%
# HuggingFace model name

lora_path = model_path / "speech-lora"
# Create the LLM object with LoRA adaptation
llm = LLM(
    model=str(model_path),
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
    enable_lora=True,
    limit_mm_per_prompt={"audio": 1},
)

prompts = [
    "<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"
]
audios = ["/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
# prompts = ["Hello, AI!", "Tell me a joke"]
# Model paths and processor
# Example prompt
prompt = "What is the capital of France?"
# Set sampling parameters
sampling_params = SamplingParams(temperature=1, max_tokens=640)
# Generate output
outputs = llm.generate([prompt], sampling_params)
# Print the generated text
for output in outputs:
    print(output.outputs[0].text)
# %%
