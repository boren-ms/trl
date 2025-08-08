# %%
import os
from vllm import LLM, SamplingParams
from trl.data_utils import sf_read

os.environ["VLLM_USE_V1"] = "0"  # Disable KV cache for Whisper
# Create a Whisper encoder/decoder model instance
llm = LLM(
    model="openai/whisper-large-v3",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
    # kv_cache_dtype="fp8",
)
# %%


audio_path = "/home/boren/data/LibriSpeech/train-clean-360/115/122944/115-122944-0038.flac"
audio_data = sf_read(audio_path)

prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": audio_data,
        },
    },
    {
        "prompt": "<|startoftranscript|><|zh|><|translate|>",  # does not work yet
        "multi_modal_data": {
            "audio": audio_data,
        },
    },
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        "decoder_prompt": "<|startoftranscript|><|zh|>",
    },
]
sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=200)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for i, output in enumerate(outputs):
    print(f"[{i}]: {output.outputs[0].text}")

# %%
