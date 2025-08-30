# %%

from pathlib import Path
from trl.scripts.audio_dataset import create_audio_dataset

ds = create_audio_dataset(dataset_name="tsv", tsv_paths=["/home/boren/data/LibriSpeech/debug.tsv"])


print(ds[0])

# %%
egs = ds[0]
import random

prefix_ratio = [0.0, 0.3]

prompt_format = "<|user|><|audio_1|>{}<|end|><|assistant|>"


def complete_string(egs):
    words = egs["text"].split()
    ratio = random.uniform(prefix_ratio[0], prefix_ratio[-1])
    n_pfx = int(len(words) * ratio)
    prefix = " ".join(words[:n_pfx])
    prompt = f"Transcribe the audio clip into text with the prefix [{prefix}]"
    return {
        "text": " ".join(words[n_pfx:]),
        "prompt": prompt_format.format(prompt),
    }


ds = ds.map(complete_string)

for x in ds:
    print(x["prompt"])
    print(x["text"])


# %%
