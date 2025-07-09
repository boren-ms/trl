# %%
import torch
import soundfile
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


# model_path = str(Path(__file__).parent)
# model_path = "/root/data/ckp/hf_models/phi4_mm_bias_merged"
home_dir = Path.home()
model_path = f"{home_dir}/data/hf_models/phi4_mm_bias_merged"
print(f"model_path: {model_path}")
assert Path(model_path).exists(), f"Model path {model_path} does not exist."
kwargs = {}
kwargs["torch_dtype"] = torch.bfloat16
device = "cuda:0"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# print(processor.tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path.rstrip("/"),
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
).to(device)
print("model.config._attn_implementation:", model.config._attn_implementation)
# %%

generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")

# %%
user_prompt = "<|user|>"
assistant_prompt = "<|assistant|>"
prompt_suffix = "<|end|>"

wav_paths = [f"{home_dir}/data/LibriSpeech/test-clean/2094/142345/2094-142345-0034.flac", f"{home_dir}/data/LibriSpeech/train-clean-360/115/122944/115-122944-0026.flac"]
words = ["cutlery", "utensils", "silverware", "TABLE", "CLOTHS", "Napkins", "Linen", "dining"]
words_text = ", ".join([f"*{w}*" for w in words])
text = "Transcribe the audio clip into text."
text = f"{text} Please pay attention to following words: {words_text}."
inputs = []
N = 10
n_gen = 4
#%%
prompt =[ {
    "role": "user",
    "content": f"<|audio_1|> Transcribe the audio clip into text.",
}]
prompt_text = processor.tokenizer.apply_chat_template(
   prompt,
    tokenize=False,
    add_generation_prompt=True
)
print (f">>> Prompt\n{prompt}")
print(f">>> Prompt Text\n{prompt_text}")


# %%
for wav_path in wav_paths[:N]:

    # inputs.append(
    #     {
    #         "prompt": f"<|user|><|audio_1|>{text}<|end|><|assistant|>",
    #         "multi_modal_data": {"audio": [sf.read(wav_path)]},
    #     }
    # )
    ########################## speech only ################################
    # speech_prompt = "Please transcribe the audio clip into text. Please pay attention to the following words: *hoped*, *stew*, *dinner*, *turnips*, *carrots*, *bruised*, *potatoes*, *mutton*, *pieces*, *ladled*, *thick*, *peppered*, *flour*, *fatten*, *sauce*."

    prompt = f"{user_prompt}<|audio_1|>{text}{prompt_suffix}{assistant_prompt}"
        prompts_text = [
            self.processing_class.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    print(f">>> Prompt\n{prompt}")
    print(">>>Wav Path:", wav_path)
    audio = soundfile.read(wav_path)

    inputs = processor(text=[prompt] * n_gen, audios=[audio] * n_gen, return_tensors="pt").to(device)
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.9,        
        generation_config=generation_config,
    )

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for i, res in enumerate(responses):
        print(f">>> Response {i}\n{res}")


# %%
