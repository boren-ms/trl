# %%
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from functools import partial
from biasing import PieceSampler


def bias_dataset(file_paths, num=None, ground_truth=True):
    """Create a dataset from the given split."""
    # data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
    # jsonl_path = data_dir/"test-clean.biasing_100.jsonl"
    data_files = [file_paths] if isinstance(file_paths, str) else file_paths
    data_files = [str(file_path) for file_path in data_files]
    data = load_dataset("json", data_files=data_files, split="train")
    ins_fmt = "Transcribe the audio clip into text. Please pay attention to following words: {words}."

    def load_audio(example, ground_truth=False):
        """Load audio from a file."""
        audio, sr = sf.read(example["audio_path"])
        words = example["ground_truth"] if ground_truth else example["distractors"]
        instruct = ins_fmt.format(words=", ".join(words))
        x = {
            "prompt": [{"role": "user", "content": f"<|audio_1|>{instruct}"}],
            "sr": sr,
            "audio": audio,
            "text": example["text"],
        }
        return x

    if num is not None:
        data = data.take(num)
    data = data.shuffle(seed=42)
    data = data.map(partial(load_audio, ground_truth=ground_truth))
    return data


def ls_dataset(head=None, sampler=None):
    """Create a dataset from the given split."""
    data = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        split="test.clean",
    )
    sampler_kwargs = sampler or {
        "bias_prob": 0.9,
        "hit_prob": 0.8,
        "max_piece_len": 1,
        "max_num": 10,
    }
    bias_sampler = PieceSampler(**sampler_kwargs)
    def proc_sample(sample):
        """Process a sample from the dataset."""
        context, text = bias_sampler.sample(sample["text"])
        side_prompt =  f"Please pay attention to following words: {context}." if context else ""
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": f"<|audio_1|>Transcribe the audio clip into text. {side_prompt}",
                }
            ],
            "sr": sample["audio"]["sampling_rate"],
            "audio": sample["audio"]["array"],
            "text": text,
        }

    if head is not None:
        data = data.take(head)
    data = data.map(proc_sample)
    return data


def create_dataset(name="bias", **kwargs):
    """Create a dataset from the given split."""
    if name == "bias":
        data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
        data_paths = [
            data_dir / "test-clean.biasing_100.jsonl",
            data_dir / "test-clean.biasing_500.jsonl",
            # data_dir / "test-clean.biasing_1000.jsonl",
        ]
        return bias_dataset(data_paths, **kwargs)
    elif name == "ls_bias":
        return ls_dataset(**kwargs)

    raise ValueError(f"Unknown dataset name: {name}")


# %%
if __name__ == "__main__":
    # Example usage
    dataset = create_dataset(name="bias", num=2)
    print(dataset)
    print(dataset[0]["text"])
# %%
