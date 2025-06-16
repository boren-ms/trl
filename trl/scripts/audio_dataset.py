# %%
import os
import ast
import urllib
from pathlib import Path
import random
from datasets import load_dataset, concatenate_datasets
import blobfile as bf
import soundfile as sf
from functools import partial
from error_simu import ErrorSimulator
from biasing import PieceSampler
# from trl.scripts.biasing import PieceSampler


def sf_read(file_path):
    """Load audio from a file."""
    if not bf.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with bf.BlobFile(file_path, "rb") as f:
        audio, sr = sf.read(f)
    return audio, sr


def bias_dataset(file_paths, ground_truth=True, **kwargs):
    """Create a dataset from the given split."""
    # data_dir = Path("/datablob1/users/boren/data/SR/librispeech_biasing/ref")
    # jsonl_path = data_dir/"test-clean.biasing_100.jsonl"
    data_files = [file_paths] if isinstance(file_paths, str) else file_paths
    data_files = [str(file_path) for file_path in data_files]

    ds = load_dataset(
        "json",
        data_files=data_files,
        split="train",
    )
    ds = stream_shuffle(ds, **kwargs)
    ins_fmt = "Transcribe the audio clip into text. Please pay attention to following words: {words}."

    def load_audio(example, ground_truth=False):
        """Load audio from a file."""
        audio, sr = sf_read(example["audio_path"])
        words = example["ground_truth"] if ground_truth else example["distractors"]
        instruct = ins_fmt.format(words=", ".join(words))
        x = {
            "prompt": [{"role": "user", "content": f"<|audio_1|>{instruct}"}],
            "sr": sr,
            "audio": audio,
            "text": example["text"],
        }
        return x

    ds = ds.map(partial(load_audio, ground_truth=ground_truth))
    return ds


def load_tsv(tsv_file):
    """Load a TSV file into a dataset."""
    url = urllib.parse.urlparse(tsv_file)
    options = {}
    if url.scheme == "az":  # blobfile
        options = {
            "account_name": url.netloc,
            "tenant_id": os.environ.get("AZURE_TENANT_ID"),
            "client_id": os.environ.get("AZURE_CLIENT_ID"),
            "client_secret": os.environ.get("AZURE_CLIENT_SECRET"),
        }
        # update remote path
        tsv_file = f"{url.scheme}:/{url.path}"

    ds = load_dataset(
        "csv",
        data_files=tsv_file,
        split="train",
        delimiter="\t",
        column_names=["id", "paths", "msgs"],
        storage_options=options,
    )
    
    dir_path = url._replace(path=str(Path(url.path).parent)).geturl() if url.scheme == "az" else None
    ds = ds.map(lambda x: {"dir": dir_path})
    return ds

def tsv_dataset(tsv_paths, **kwargs):
    """Create a dataset from the given split."""
    if isinstance(tsv_paths, (list, tuple)):
        ds = concatenate_datasets(
            [load_tsv(tsv_path) for tsv_path in tsv_paths]
        )
    else:
        ds = load_tsv(tsv_paths)
        
    ds = stream_shuffle(ds, **kwargs)


    def load_sample(egs):
        """Process a single sample."""
        audio_path = ast.literal_eval(egs["paths"])[0]
        if ds["dir"]:
            audio_path = audio_path.replace("/root/data/LibriSpeech/", "") # TODO: remove this line once the tsv files are updated
            audio_path = str(Path(ds["dir"]) / audio_path)
        messages = ast.literal_eval(egs["msgs"])[0]["messages"]
        audio, fs = sf_read(audio_path)
        x = {
            "audio": {
                "array": audio,
                "sampling_rate": fs,
            },
            "text": messages[-1]["content"],
            "id": egs["id"],
        }
        return x

    ds = ds.map(load_sample)
    return ds


def openasr_dataset(**kwargs):
    """Create a dataset from the given split."""
    name = kwargs.get("name", "librispeech")
    split = kwargs.get("split", "test.clean")
    ds = load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        name,
        split=split,
    )
    ds = stream_shuffle(ds, **kwargs)
    return ds


def stream_shuffle(ds, **kwargs):
    """Process the dataset."""
    streaming = kwargs.get("streaming", False)
    if streaming:
        ds = ds.to_iterable_dataset(num_shards=kwargs.get("num_shards", 1))
    num_egs = kwargs.get("num_egs", None)
    if num_egs is not None:
        ds = ds.take(num_egs)
    # dataset = dataset.shuffle(seed=42, buffer_size=kwargs.get("buffer_size", 1000))
    return ds


def bias_sampling(ds, **kwargs):
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

    ds = ds.map(proc_sample)
    return ds


def simulate_perference(ds, **kwargs):
    """simulate the perference  to the dataset."""
    error_range = kwargs.pop("error_range", (0.1, 0.25))
    if not isinstance(error_range, (tuple, list)):
        error_range = [float(error_range), float(error_range)]
    simulator = ErrorSimulator(**kwargs)

    def add_perference(sample, error_range):
        """Process a sample from the dataset."""
        err_rate = random.uniform(*error_range)
        text = sample["text"]
        bad_text = simulator.random_error(text, err_rate)
        return {
            "chosen": [
                {
                    "role": "assistant",
                    "content": text,
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": bad_text,
                }
            ],
        }

    return ds.map(add_perference, fn_kwargs={"error_range": error_range})

    # return concatenate_datasets(datasets)


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
        ds = openasr_dataset(**kwargs)
        ds = bias_sampling(ds, **kwargs.get("biasing", {}))
        ds = simulate_perference(ds, **kwargs.get("simu_perference", {}))
        return ds
    elif dataset_name == "tsv":
        ds = tsv_dataset(**kwargs)
        ds = bias_sampling(ds, **kwargs.get("biasing", {}))
        ds = simulate_perference(ds, **kwargs.get("simu_perference", {}))
        return ds
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
