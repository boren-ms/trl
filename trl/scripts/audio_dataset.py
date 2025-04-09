# %%
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from functools import partial

from trl.scripts.biasing import PieceSampler


def bias_dataset(file_paths, num=None, ground_truth=True, **kwargs):
    """Create a dataset from the given split."""
    # data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
    # jsonl_path = data_dir/"test-clean.biasing_100.jsonl"
    data_files = [file_paths] if isinstance(file_paths, str) else file_paths
    data_files = [str(file_path) for file_path in data_files]

    data = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=kwargs.get("streaming", False),
    )
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
    data = data.map(partial(load_audio, ground_truth=ground_truth))
    return data


def tsv_dataset(tsv_paths, head=None, **kwargs):
    """Create a dataset from the given split."""
    if isinstance(tsv_paths, str):
        tsv_paths = [tsv_paths]
    data = load_dataset(
        "csv",
        data_files=[str(tsv_path) for tsv_path in tsv_paths],
        split="train",
        delimiter="\t",
        column_names=["id", "paths", "msgs"],
        streaming=kwargs.get("streaming", False),
    )

    def load_sample(egs):
        """Process a single sample."""
        audio_path = eval(egs["paths"])[0]
        messages = eval(egs["msgs"])[0]["messages"]
        audio, fs = sf.read(audio_path)
        x = {
            "audio": {
                "array": audio,
                "sampling_rate": fs,
            },
            "text": messages[-1]["content"],
            "id": egs["id"],
        }
        return x

    if head is not None:
        data = data.take(head)
    data = data.map(load_sample)
    return data


def openasr_dataset(head=None, **kwargs):
    """Create a dataset from the given split."""
    name = kwargs.get("name", "librispeech")
    split = kwargs.get("split", "test.clean")
    dataset = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        name,
        split=split,
        streaming=kwargs.get("streaming", False),
    )
    if head is not None:
        dataset = dataset.take(head)
    return dataset


def apply_bias_sampling(dataset, **kwargs):
    """Apply bias sampling to the dataset."""
    kwargs = kwargs or {
        "bias_prob": 0.9,
        "hit_prob": 0.9,
        "max_piece_len": 1,
        "max_num": 2,
    }
    bias_sampler = PieceSampler(**kwargs)

    def proc_sample(sample):
        """Process a sample from the dataset."""
        context, text = bias_sampler.sample(sample["text"])
        side_prompt = (
            f"Please pay attention to following words: {context}." if context else ""
        )
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

    dataset = dataset.map(proc_sample)
    return dataset


def create_dataset(name="openasr", **kwargs):
    """Create a dataset from the given split."""
    if name == "ls_bias":
        data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
        data_paths = [
            data_dir / "test-clean.biasing_100.jsonl",
            data_dir / "test-clean.biasing_500.jsonl",
            # data_dir / "test-clean.biasing_1000.jsonl",
        ]
        return bias_dataset(data_paths, **kwargs)
    elif name == "openasr":
        dataset = openasr_dataset(**kwargs)
        dataset = apply_bias_sampling(dataset, **kwargs.get("biasing", {}))
        return dataset
    elif name == "tsv":
        dataset = tsv_dataset(**kwargs)
        dataset = apply_bias_sampling(dataset, **kwargs.get("biasing", {}))
        return dataset
    raise ValueError(f"Unknown dataset name: {name}")


# %%
if __name__ == "__main__":
    # Example usage
    # dataset = create_dataset(name="openasr", head=2)
    # print(dataset)
    # print(dataset[0]["text"])
    tsv_path = "/datablob1/users/ruchaofan/wavllm_data/wavllm/converted_path_train_data_4chunk/asr_train_transcribe.tsv"
    dataset = create_dataset(name="tsv", head=2, tsv_paths=[tsv_path])
    print(dataset)
    print(dataset[0]["text"])

# %%
