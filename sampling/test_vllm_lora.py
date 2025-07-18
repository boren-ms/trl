# %%
import os
from typing import Any, NamedTuple, Optional
from vllm.lora.request import LoRARequest
import vllm
from pathlib import Path
from vllm import LLM, SamplingParams
import torch
from transformers import GenerationConfig, AutoProcessor
from vllm import LLM, SamplingParams
from trl.data_utils import sf_read
from datasets import load_dataset  # Added import


# %%


# model_path = "/root/data/ckp/hf_models/phi4_mm_bias"
model_path = Path("/root/data/ckp/hf_models/Phi-4-multimodal-instruct")

# %%
# HuggingFace model name

speech_lora_path = model_path / "speech-lora"
# Create the LLM object with LoRA adaptation
llm = LLM(
    model=str(model_path),
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
    enable_lora=True,
    limit_mm_per_prompt={"audio": 2},
    max_lora_rank=320,
)
#%%
# prompt = "What is the capital of France?"
# outputs = llm.generate([prompt], SamplingParams(temperature=1, max_tokens=640))
# for output in outputs:
#     print(output.outputs[0].text)
# %%

prompts = [
    "<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"
]
audios = ["/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
inputs = [
        {"prompt": prompt, "multi_modal_data": {"audio": [sf_read(audio_path)]}}
        for prompt, audio_path in zip(prompts, audios)
]
sampling_params = SamplingParams(temperature=1, max_tokens=640)
#%%
lora_request = [LoRARequest("speech", 1, str(speech_lora_path))]
outputs = llm.generate(inputs, sampling_params=sampling_params,  lora_request=lora_request)
texts = [output.outputs[0].text for output in outputs]
text = texts[0]
print(text)
# %%
