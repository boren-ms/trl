# %%
import torch
from transformers import GenerationConfig, AutoProcessor
from vllm import LLM, SamplingParams
from trl.data_utils import sf_read
from datasets import load_dataset  # Added import


#%%
import os
os.environ["RANK"]= "0"
os.environ["LOCAL_RANK"]= "0"
os.environ["WORLD_SIZE"]= "1"
os.environ["MASTER_ADDR"]= "localhost"
os.environ["MASTER_PORT"]= "12355"

#%%
#%%

def hf2vllm_config(hf_config):
    """Convert a HuggingFace GenerationConfig or dict to vLLM sampling parameters."""
    mapping = {
        "max_new_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "num_return_sequences": "n",
        "eos_token_id": "stop_token_ids",
    }
    vllm_params = {vllm_key: hf_config[hf_key] for hf_key, vllm_key in mapping.items() if hf_key in hf_config and hf_config[hf_key] is not None}
    if "do_sample" in hf_config and not hf_config["do_sample"]:
        vllm_params["temperature"] = 0.0
    vllm_params["n"] = vllm_params.get("n", 1)
    return vllm_params


#%%
model_path="/root/data/ckp/hf_models/phi4_mm_bias_merged"

generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
config = hf2vllm_config(generation_config.to_dict())
#%%
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    dtype=torch.float32,
    max_model_len=1024 * 6, # this will 5*1024 still to short
    distributed_executor_backend="external_launcher",
    seed=0,
    max_num_seqs=128,
    load_format="auto",
    limit_mm_per_prompt={"audio": 1},
)
#%%
jsonl_path="/root/data/ckp/hf_models/phi4_mm_bias_merged/clean_1000_hf_vllm_diff.jsonl"
dataset = load_dataset("json", data_files=jsonl_path, split="train")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
#%%

data_dir ="/root/data/LibriSpeech/test-clean/"

for example in dataset:
    id = example["id"]
    if id != "5105-28233-0007":
        continue
    rel_dir = "/".join(id.split("-")[:2])
    audio_path = f"{data_dir}/{rel_dir}/{id}.flac"
    inputs = [
        {"prompt": example["prompt"], "multi_modal_data": {"audio": [sf_read(audio_path)]}}
    ]
    hf_x = processor(text=[example["prompt"]], audios=[sf_read(audio_path)], return_tensors="pt")
    print("input token length:", hf_x["input_ids"].shape[1])
    config["max_tokens"] = 512
    config["ignore_eos"] = True
    config["n"]=1
    sampling_params = SamplingParams(**config)
    # sampling_params.stop_token_ids = [sampling_params.stop_token_ids[1]]
    print(sampling_params)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    texts = [output.outputs[0].text for output in outputs]
    text = texts[0]
    print("ID       :", id)
    print("hyp_vlLM :", example["hyp_vllm"])
    print("HYP_HF   :", example["hyp_hf"])
    for i, text in enumerate(texts):
        print(f"[{i}]TEXT   :", text)
    break
    
#%%

# %%
for example in dataset.take(10):
    id = example["id"]
    print("ID       :", id)
    print("HYP_HF   :", example["hyp_hf"])
    print("hyp_vlLM :", example["hyp_vllm"])
    print("*"*20)    
# %%
id = "5105-28233-0007"
hyp_hf = "Ben *zoof's* most *ambitious* desire was to *induce* the captain to go with him and end his days in his much loved home, and so *incessantly* were *servadac's* ears *besieged* with *descriptions* of the *unparalleled* *beauties* and advantages of this *eighteenth* *arrondissement* of Paris that he could scarcely hear the name of *montmartre* without a conscious *thrill* of *aversion*"

rel_dir = "/".join(id.split("-")[:2])
audio_path = f"{data_dir}/{rel_dir}/{id}.flac"
inputs = [
    {"prompt": example["prompt"], "multi_modal_data": {"audio": [sf_read(audio_path)]}}
]
config["max_tokens"] = 512
sampling_params = SamplingParams(**config)
print("Sampling Param:", sampling_params)
outputs = llm.generate(inputs, sampling_params=sampling_params)
texts = [output.outputs[0].text for output in outputs]
text = texts[0]
print(f"ID     : {id}")
print(f"TEXT   : {texts[0]}")
print(f"HF_HYP : {hyp_hf}")

# %%
