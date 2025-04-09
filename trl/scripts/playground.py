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
jsonl_path = data_dir / "test-clean.biasing_100.jsonl"
dataset = load_dataset("json", data_files=str(jsonl_path), split="train")
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
# %%
from functools import partial

newdata = dataset.map(
    partial(load_audio, ground_truth=True),
    remove_columns=["audio_path", "text", "distractors", "ground_truth"],
)
# %%
from difflib import SequenceMatcher
import re


def get_align(reference, hypothesis, btag="*"):
    """Aligns the reference and hypothesis strings and returns the alignment details."""
    refs = reference.split()
    hyps = hypothesis.split()
    matcher = SequenceMatcher(
        None, [x.strip(btag) for x in refs], [x.strip(btag) for x in hyps]
    )
    alignment = []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        alignment.append((operation, " ".join(refs[i1:i2]), " ".join(hyps[j1:j2])))
    return alignment


hyp = "have you not met it *anywhere*"
ref = "*have* *you* *not* *met them* *anywhere*"

for tag, ref_part, hyp_part in get_align(ref, hyp):
    print(f"{tag}: '{ref_part}' -> '{hyp_part}'")
# %%


def count_starred_phrases(text):
    """Counts the number of phrases surrounded by '*' in the given text."""
    return len(re.findall(r"\*.*?\*", text))


text = "*have* *you* *not* *met them* *anywhere*"
count = count_starred_phrases(text)
print(f"Number of starred phrases: {count}")
# %%
import jiwer.transforms as tr


class RemovePunctuationExclude(tr.RemovePunctuation):
    """RemovePunctuation excluding certain characters."""

    def __init__(self, exclude=None):
        super().__init__()
        self.exclude = exclude or []
        self.tokens_to_remove = [
            x for x in self.tokens_to_remove if x not in self.exclude
        ]
        print(f"tokens_to_remove: {self.tokens_to_remove}")


norm = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.ExpandCommonEnglishContractions(),
        # tr.RemovePunctuation(),
        RemovePunctuationExclude(exclude=["*"]),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ReduceToSingleSentence(),
    ]
)
text = "*have* *you'll* *not,* *met, them* *anywhere*"
print(text)
print(norm(text))
# %%

from datasets import load_dataset
import fsspec

# Set a longer timeout (e.g., 60 seconds)
fsspec.config.conf["timeout"] = 600

cache_dir = "/datablob1/users/boren/data/librispeech_asr"
data = load_dataset(
    "openslr/librispeech_asr", trust_remote_code=True, cache_dir=cache_dir
)
# %%
from datasets import load_dataset

data = load_dataset(
    "openslr/librispeech_asr",
    "clean",
    data_dir="/datablob1/users/boren/data/librispeech_asr/train-clean-100",
    split="train.100",
    trust_remote_code=True,
)
# %%
from pathlib import Path
from datasets import load_dataset


tsv_path = Path(
    "/datablob1/users/ruchaofan/wavllm_data/wavllm/converted_path_train_data_4chunk/asr_train_transcribe.tsv"
)

# Load TSV dataset with specified column names
data = load_dataset(
    "csv",
    data_files=[str(tsv_path)],
    split="train",
    delimiter="\t",
    column_names=["id", "paths", "messages"],
)


def proc_sample(example):
    """Process a single sample."""
    audio_path = eval(example["paths"])[0]
    messages = eval(example["messages"])[0]["messages"]
    audio, fs = sf.read(audio_path)
    x = {
        "audio": audio,
        "sr": fs,
        "text": messages[-1]["content"],
        "id": example["id"],
    }
    return x


print(data)


class TsvDataset(dataset):
    def __init__(self, file_paths, num=None):
        self.file_paths = file_paths
        self.num = num

    def load_audio(self, example):
        """Load audio from a file."""
        audio_path = example["paths"]
        audio, sr = sf.read(audio_path)
        instruct = example["messages"]
        x = {
            "prompt": [{"role": "user", "content": f"<|audio_1|>{instruct}"}],
            "sr": sr,
            "audio": audio,
            "text": example["text"],
        }
        return x

    def load_dataset(self):
        data = load_dataset(
            "csv",
            data_files=self.file_paths,
            split="train",
            delimiter="\t",
            column_names=["id", "paths", "messages"],
        )
        if self.num is not None:
            data = data.take(self.num)
        data = data.shuffle(seed=42)
        data = data.map(self.load_audio)
        return data


# %%
import pytz
from datetime import datetime

tz = pytz.timezone("America/Los_Angeles")  # UTC-7/UTC-8 depending on DST
print(f"{datetime.now(tz).strftime('%Y%m%d-%H%M%S')}")

# %%
