#%%
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora.layer import LoraLayer

# %%# LoRA related settings



def clone_phimm_lora(model,  dst_name, src_name="speech"):
    """Clone the LoRA adapter from src_name to dst_name."""
    print(f"Cloning LoRA adapter from {src_name} to {dst_name}")
    src_lora_config = getattr(model.config, f"{src_name}_lora")
    lora_conf = LoraConfig(
        r=src_lora_config['r'],
        lora_alpha=src_lora_config['lora_alpha'],
        # target_modules=src_lora_config['layer'].replace("layers.", "model.layers."),
        target_modules=src_lora_config['layer'],
        lora_dropout=src_lora_config['dp'],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model.model, lora_conf, adapter_name=dst_name)
    for module in peft_model.modules():
        if not isinstance(module, LoraLayer):
            continue
        if module.merged:
            module.unmerge()
        module.lora_A[dst_name].weight.data = module.lora_A[src_name].weight.data
        module.lora_B[dst_name].weight.data = module.lora_B[src_name].weight.data
        module.set_adapter(dst_name)
        module._disable_adapters = False
    
    return peft_model
    
# %%
model_id =  "microsoft/Phi-4-multimodal-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)
model.set_lora_adapter("speech")
#%%
peft_model = clone_phimm_lora(model, "speech_biasing")
#%%
# List the training parameters of the LoRA adapter and count the total
total_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
        total_params += param.numel()

print(f"Total trainable parameters: {total_params:,}")