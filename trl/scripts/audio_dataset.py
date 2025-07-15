# %%
import os
import ast
import urllib
import random
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from trl.scripts.error_simu import ErrorSimulator
from trl.scripts.biasing import PieceSampler, tag_pieces, text_norm
from trl.data_utils import sf_read

prompt_format = "<|user|><|audio_1|>{}<|end|><|assistant|>"


def stream_shuffle(ds, **kwargs):
    """Process the dataset."""
    streaming = kwargs.get("streaming", False)
    if streaming:
        ds = ds.to_iterable_dataset(num_shards=kwargs.get("num_shards", 1))
    num_egs = kwargs.get("num_egs", None)
    if num_egs is not None:
        ds = ds.take(num_egs)
    return ds


def ls_bias_dataset(jsonl_path, bias_key=None, tag="*", data_dir=None, **kwargs):
    """Create a dataset from the given split."""
    data_files = [jsonl_path] if isinstance(jsonl_path, str) else jsonl_path
    data_files = [str(file_path) for file_path in data_files]

    ds = load_dataset(
        "json",
        data_files=data_files,
        split="train",
    )

    ds = stream_shuffle(ds, **kwargs)

    def load_sample(example):
        """Load audio from a file."""

        bias_words = example.get(bias_key, [])
        bias_str = ", ".join(tag_pieces(bias_words, tag=tag))
        instruct = "Transcribe the audio clip into text."
        if bias_str:
            instruct += f" Pay extra attention to the following phrases/words: {bias_str}."
        audio_path = Path(example["audio_path"])
        idx = audio_path.stem
        if data_dir:
            if audio_path.is_absolute():
                audio_path = audio_path.relative_to("/root/data")
            audio_path = f"{data_dir}/{audio_path}"  # not use Path here, since it may be a remote path
        words = example.get("text", "").strip().split()
        gt_words = example.get("ground_truth", [])
        words = tag_pieces(words, tag=tag, specified=gt_words, norm=text_norm)
        return {
            "prompt": prompt_format.format(instruct),
            "audio_path": str(audio_path),
            "text": " ".join(words),
            "id": example.get("id", idx),
        }

    ds = ds.map(load_sample)
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
    print("DATA DIR:", dir_path)
    ds = ds.map(lambda x: {"dir": dir_path})
    return ds


def tsv_dataset(tsv_paths, **kwargs):
    """Create a dataset from the given split."""
    if isinstance(tsv_paths, (list, tuple)):
        ds = concatenate_datasets([load_tsv(tsv_path) for tsv_path in tsv_paths])
    else:
        ds = load_tsv(tsv_paths)

    ds = stream_shuffle(ds, **kwargs)

    def load_sample(egs):
        """Process a single sample."""
        audio_path = ast.literal_eval(egs["paths"])[0]
        if egs["dir"]:
            audio_path = audio_path.replace("/root/data/LibriSpeech", egs["dir"])
        messages = ast.literal_eval(egs["msgs"])[0]["messages"]
        x = {
            "audio_path": audio_path,
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


def post_process(ds, **kwargs):
    """Post process the dataset."""
    load_audio = kwargs.get("load_audio", False)
    def read_audio(sample):
        """Read audio from the file."""
        audio, sr = sf_read(sample["audio_path"])
        return {"audio": audio, "sr": sr}
    cols = ["prompt", "text", "audio_path", "id"]
    if load_audio:
        cols += ["audio", "sr"]
        ds = ds.map(read_audio)
    # select required only
    ds = ds.select_columns(cols)
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
        side_prompt = f"Pay extra attention to the following phrases/words: {context}." if context else ""
        return {
            "prompt": prompt_format.format(f"Transcribe the audio clip into text. {side_prompt}"),
            "text": text,  # text is updated
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



def create_audio_dataset(dataset_name="openasr", **kwargs):
    """Create a dataset from the given split."""
    ds = None
    if dataset_name == "ls_bias":
        ds = ls_bias_dataset(**kwargs)
    elif dataset_name == "openasr":
        ds = openasr_dataset(**kwargs)
        ds = bias_sampling(ds, **kwargs.get("biasing", {}))
        ds = simulate_perference(ds, **kwargs.get("simu_perference", {}))
    elif dataset_name == "tsv":
        ds = tsv_dataset(**kwargs)
        ds = bias_sampling(ds, **kwargs.get("biasing", {}))
        ds = simulate_perference(ds, **kwargs.get("simu_perference", {}))
    if ds:
        return post_process(ds, **kwargs)
    raise ValueError(f"Unknown dataset name: {dataset_name}")


# %%
if __name__ == "__main__":
    # Example usage
    # dataset = create_dataset(name="openasr", head=2)
    # print(dataset)
    # print(dataset[0]["text"])
    tsv_path = "/datablob1/users/ruchaofan/wavllm_data/wavllm/converted_path_train_data_4chunk/asr_train_transcribe.tsv"
    dataset = create_audio_dataset(dataset_name="tsv", num_egs=2, tsv_paths=[tsv_path])
    print(dataset)
    print(next(iter(dataset)))

# %%
