# %%
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
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
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)
stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt").input_ids.flatten().tolist()

# %%
tsv_path = "/root/data/LibriSpeech/debug.tsv"
wav_paths = load_wav_path(tsv_path)

# %%
wav_paths = ["/root/data/LibriSpeech/test-clean/2094/142345/2094-142345-0034.flac", "/root/data/LibriSpeech/train-clean-360/115/122944/115-122944-0026.flac"]
words = ["cutlery", "utensils", "silverware", "TABLE", "CLOTHS", "Napkins", "Linen", "dining"]
words_text = ", ".join([f"*{w}*" for w in words])
text = "Transcribe the audio clip into text."
text = f"{text} Please pay attention to following words: {words_text}."
inputs = []
N = 10
for wav_path in wav_paths[:N]:

    inputs.append(
        {
            "prompt": f"<|user|><|audio_1|>{text}<|end|><|assistant|>",
            "multi_modal_data": {"audio": [sf.read(wav_path)]},
        }
    )


# %%
stop = ["<|end|>", "<|endoftext|>"]
n_gen = 8
outputs = llm.generate(
    inputs,
    # sampling_params=SamplingParams(temperature=1, max_tokens=512, n=n_gen, stop=stop),
    sampling_params=SamplingParams(temperature=1, max_tokens=512, n=n_gen, stop_token_ids=stop_tokens_ids),
)
for i, o in zip(inputs, outputs):
    print(f"Utt: {i['prompt']}")
    for j, output in enumerate(o.outputs):
        print(f"[{j}]:", output.text)
# %%
