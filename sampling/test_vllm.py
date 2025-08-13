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
model_path = Path("/home/boren/data/ckp/hf_models/Phi-4-multimodal-instruct")

# %%
# HuggingFace model name

speech_lora_path = model_path / "speech-lora"
# Create the LLM object with LoRA adaptation
llm = LLM(
    model=str(model_path),
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=1,
    logprobs=1,
    # enable_lora=True,
    limit_mm_per_prompt={"audio": 2},
    # max_lora_rank=320,
)
# %%
# prompt = "What is the capital of France?"
# outputs = llm.generate([prompt], SamplingParams(temperature=1, max_tokens=640))
# for output in outputs:
#     print(output.outputs[0].text)
# %%
sampling_params = SamplingParams(temperature=1, max_tokens=8192)
prompts = ["<|user|><|audio_1|>Transcribe the audio clip into text. <|end|><|assistant|>"]
audios = ["/home/boren/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"]
inputs = [{"prompt": prompt, "multi_modal_data": {"audio": [sf_read(audio_path)]}} for prompt, audio_path in zip(prompts, audios)]

# %%
outputs = llm.generate(inputs, sampling_params=sampling_params)
texts = [output.outputs[0].text for output in outputs]
text = texts[0]
print(text)
# %%
text_prompt = "Please capture the text and output it in <result> <text></text>"
text_prompt = "Transcribe the audio clip into text. the text in  <text></text> format. For example, <text>Paris</text>."
inputs[0]["prompt"] = f"<|user|><|audio_1|>{text_prompt}<|end|><|assistant|>"
sampling_params = SamplingParams(temperature=1, max_tokens=8192)
lora_request = [LoRARequest("speech", 1, str(speech_lora_path))]
outputs = llm.generate(inputs, sampling_params=sampling_params, lora_request=lora_request)
texts = [output.outputs[0].text for output in outputs]
text = texts[0]
print(text)
# %%
