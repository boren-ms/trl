# %%
from datasets import load_dataset

dataset_path = "hf-audio/esb-datasets-test-only-sorted"
dataset = "librispeech"
split = "test.clean"
data = load_dataset(dataset_path, dataset, split=split)
print(len(data))

# %%
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from functools import partial
data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
jsonl_path = data_dir/"test-clean.biasing_100.jsonl"
dataset = load_dataset("json",data_files=str(jsonl_path), split="train")
print(dataset)

# INSTRUCTION = "Transcribe the audio clip into text."
INSTRUCTION = "Transcribe the audio clip into text. Please pay attention to following words: {words}."

def load_audio(example, ground_truth=False):
    """Load audio from a file."""
    audio, sr = sf.read(example["audio_path"])
    words = example["ground_truth"] if ground_truth else example["distractors"]
    instruct = INSTRUCTION.format(words=", ".join(words))
    x = {
        "prompt": [{"role": "user", "content": f"<|audio_1|>{instruct}"}],
        "sr": sr,
        "audio": audio,
        "completion": example["text"],
    }
    return x
  
dataset = dataset.take(2)
#%%
from functools import partial

newdata = dataset.map(partial(load_audio, ground_truth=True), remove_columns=["audio_path", "text", "distractors", "ground_truth"])
# %%
