# %%
def hf2vllm_config(hf_config, n=None):
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
    if n is not None:
        vllm_params["n"] = n
    return vllm_params
