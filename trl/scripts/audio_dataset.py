# %%
from pathlib import Path
from datasets import load_dataset
import soundfile as sf
from functools import partial

from biasing import PieceSampler

# from trl.scripts.biasing import PieceSampler


def bias_dataset(file_paths, ground_truth=True, **kwargs):
    """Create a dataset from the given split."""
    # data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
    # jsonl_path = data_dir/"test-clean.biasing_100.jsonl"
    data_files = [file_paths] if isinstance(file_paths, str) else file_paths
    data_files = [str(file_path) for file_path in data_files]

    dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
    )
    dataset = stream_shuffle(dataset, **kwargs)
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

    dataset = dataset.map(partial(load_audio, ground_truth=ground_truth))
    return dataset


def tsv_dataset(tsv_paths, **kwargs):
    """Create a dataset from the given split."""
    if isinstance(tsv_paths, str):
        tsv_paths = [tsv_paths]
    dataset = load_dataset(
        "csv",
        data_files=[str(tsv_path) for tsv_path in tsv_paths],
        split="train",
        delimiter="\t",
        column_names=["id", "paths", "msgs"],
    )
    dataset = stream_shuffle(dataset, **kwargs)

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

    dataset = dataset.map(load_sample)
    return dataset


def openasr_dataset(**kwargs):
    """Create a dataset from the given split."""
    name = kwargs.get("name", "librispeech")
    split = kwargs.get("split", "test.clean")
    dataset = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        name,
        split=split,
    )
    dataset = stream_shuffle(dataset, **kwargs)
    return dataset


def stream_shuffle(dataset, **kwargs):
    """Process the dataset."""
    # dataset = dataset.to_iterable_dataset(num_shards=kwargs.get("num_shards", 128))
    num_egs = kwargs.get("num_egs", None)
    if num_egs is not None:
        dataset = dataset.take(num_egs)
    # dataset = dataset.shuffle(seed=42, buffer_size=kwargs.get("buffer_size", 1000))
    return dataset


def bias_sampling(dataset, **kwargs):
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


def create_dataset(dataset_name="openasr", **kwargs):
    """Create a dataset from the given split."""
    if dataset_name == "ls_bias":
        data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
        data_paths = [
            data_dir / "test-clean.biasing_100.jsonl",
            data_dir / "test-clean.biasing_500.jsonl",
            # data_dir / "test-clean.biasing_1000.jsonl",
        ]
        return bias_dataset(data_paths, **kwargs)
    elif dataset_name == "openasr":
        dataset = openasr_dataset(**kwargs)
        dataset = bias_sampling(dataset, **kwargs.get("biasing", {}))
        return dataset
    elif dataset_name == "tsv":
        dataset = tsv_dataset(**kwargs)
        dataset = bias_sampling(dataset, **kwargs.get("biasing", {}))
        return dataset
    raise ValueError(f"Unknown dataset name: {dataset_name}")


# %%
if __name__ == "__main__":
    # Example usage
    # dataset = create_dataset(name="openasr", head=2)
    # print(dataset)
    # print(dataset[0]["text"])
    tsv_path = "/datablob1/users/ruchaofan/wavllm_data/wavllm/converted_path_train_data_4chunk/asr_train_transcribe.tsv"
    dataset = create_dataset(dataset_name="tsv", num_egs=2, tsv_paths=[tsv_path])
    print(dataset)
    print(next(iter(dataset)))

# %%
