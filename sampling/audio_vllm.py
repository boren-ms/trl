# %%
from vllm import LLM, SamplingParams
import soundfile as sf
import pandas as pd


def load_wav_path(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", names=["id", "wav_paths", "msgs"])
    df["wav_path"] = df["wav_paths"].apply(lambda x: eval(x)[0])
    return df["wav_path"].tolist()


# %%
model_path = "/root/data/ckp/hf_models/phi4_mm_bias_merged"
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    max_model_len=512,
    max_num_seqs=8,
    load_format="auto",
    limit_mm_per_prompt={"audio": 10},
)
# %%
tsv_path = "/root/data/LibriSpeech/debug.tsv"
wav_paths = load_wav_path(tsv_path)

# wav_path = "/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0026.flac"
question = "Transcribe the audio into text."
inputs = []
N = 10
for wav_path in wav_paths[:N]:
    inputs.append(
        {
            "prompt": f"<|user|><|audio_1|>{question}<|end|><|assistant|>",
            "multi_modal_data": {"audio": [sf.read(wav_path)]},
        }
    )


# %%
# TODO: need to add the stop_token from model to ensure the model stops at the right place.
sampling_params = SamplingParams(temperature=0.2, max_tokens=512, n=2)
outputs = llm.generate(inputs, sampling_params=sampling_params)
# %%
for i, o in enumerate(outputs):
    print(f"Utt {i}")
    for j, output in enumerate(o.outputs):
        print(f"[{j}]:", output.text)
# %%
