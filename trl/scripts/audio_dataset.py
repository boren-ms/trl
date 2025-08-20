# %%
import os
import ast
import urllib
import random
import blobfile as bf
import pandas as pd
import numpy as np
from functools import partial
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from bs4 import BeautifulSoup
from trl.scripts.error_simu import ErrorSimulator
from trl.scripts.biasing import PieceSampler, tag_pieces, text_norm
from trl.scripts.audio_prompts import get_task_prompt
from trl.scripts.chunk_dataset import generate_examples, get_chunk_manager
from trl.data_utils import sf_read


prompt_format = "<|user|><|audio_1|>{}<|end|><|assistant|>"


def extract_entities(text):
    """extract named entities from text that are surrounded by <NE> </NE> or <NE:type> </NE:type> tags."""

    bs = BeautifulSoup(text, "html.parser")
    entities = [tag.get_text().strip() for tag in bs.find_all() if tag.name.startswith("ne")]
    return set(entities)


def jsonl_dataset(jsonl_paths, **kwargs):
    """Load a JSONL dataset from the specified paths."""

    data_files = [jsonl_paths] if isinstance(jsonl_paths, str) else jsonl_paths
    data_files = [str(file_path) for file_path in data_files]
    options = {}
    url = urllib.parse.urlparse(data_files[0])
    if url.scheme == "az":  # blobfile
        account_name = url.netloc
        options = {
            "account_name": account_name,
            "tenant_id": os.environ.get("AZURE_TENANT_ID"),
            "client_id": os.environ.get("AZURE_CLIENT_ID"),
            "client_secret": os.environ.get("AZURE_CLIENT_SECRET"),
        }
        data_files = [file.replace(f"{account_name}/", "") for file in data_files]
    ds = load_dataset("json", data_files=data_files, split="train", storage_options=options)
    ds = stream_shuffle(ds, **kwargs)
    return ds


def update_dir(data_path, src_dir=None, dst_dir=None):
    if not src_dir or not dst_dir:
        return data_path
    data_path = str(data_path)
    src_dir = src_dir.rstrip("/") + "/"  # Ensure src_dir is a clean path
    dst_dir = dst_dir.rstrip("/") + "/"  # Ensure dst_dir is a clean path
    return data_path.replace(src_dir, dst_dir) if data_path.startswith(src_dir) else data_path


def ls_bias_dataset(jsonl_path, bias_key=None, tag="*", data_dir=None, **kwargs):
    """Create a dataset from the given split."""
    ds = jsonl_dataset(jsonl_path, **kwargs)

    def load_sample(example):
        """Load audio from a file."""
        bias_words = example.get(bias_key, [])
        bias_str = ", ".join(tag_pieces(bias_words, tag=tag))
        prompt = get_task_prompt(task="biasing" if bias_str else "asr")
        audio_path = update_dir(example["audio_path"], src_dir="/root/data", dst_dir=data_dir)
        words = example.get("text", "").strip().split()
        gt_words = example.get("ground_truth", [])
        words = tag_pieces(words, tag=tag, specified=gt_words, norm=text_norm)
        return {
            "prompt": prompt_format.format(f"{prompt} {bias_str}"),
            "audio_path": audio_path,
            "text": " ".join(words),
            "keywords": gt_words,
            "id": example.get("id", Path(audio_path).stem),
        }

    ds = ds.map(load_sample)
    return ds


def read_words(file_path):
    if not file_path:
        return []
    if not bf.exists(file_path):
        return []
    with bf.BlobFile(file_path, "r") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def chunk_dataset(specs, chunk_types=None, chunk_shuffle=True, max_chunks=None, max_egs=None, streaming=False, max_cached_chunk=None, **kwargs):
    """Iterate over the chunk dataset based on the specification files."""
    if max_cached_chunk is not None:
        get_chunk_manager(max_cached_chunk)  # Initialize the chunk manager with a maximum size. and reuse later.
    gen = partial(generate_examples, specs, chunk_types, chunk_shuffle, max_chunks, max_egs)
    if streaming:
        print("Creating streaming chunk dataset.")
        ds = Dataset.from_generator(gen)
    else:
        print("Creating non-streaming chunk dataset, please be patient.")
        ds = Dataset.from_list(list(gen()))
        print(f"Loaded {len(ds)} examples from chunk dataset.")
    ds = ds.rename_column("transcription", "text")
    return ds


def entity_dataset(jsonl_path, max_bias=0, entity_file=None, distractor_file=None, tag="*", src_dir=None, data_dir=None, **kwargs):
    ds = jsonl_dataset(jsonl_path, **kwargs)
    distractors = read_words(distractor_file)
    shared_entities = read_words(entity_file)

    def load_sample(example):
        """Load audio from a file."""
        nonlocal src_dir  # not a local variable

        trans = example.get("Transcription", "").strip()
        src_dir = src_dir or "/datablob1/users/ruchaofan"
        audio_path = update_dir(example["WavPath"], src_dir=src_dir, dst_dir=data_dir)
        bs = BeautifulSoup(trans, "html.parser")

        entities = [tag.get_text().strip() for tag in bs.find_all() if tag.name.startswith("ne")]
        entities = list(set(entities + shared_entities))  # Combine with shared entities

        utt_id = example.get("UUID", Path(audio_path).stem)

        if max_bias > 0 and max_bias < len(entities):
            print(f"Groundtruth words [{len(entities)}] exceed max_bias [{max_bias}], truncating.")
        bias_words = entities.copy()[:max_bias]
        bias_words += distractors[: max(0, max_bias - len(bias_words))]

        bias_str = ", ".join(tag_pieces(bias_words, tag=tag))
        prompt = get_task_prompt(task="biasing" if bias_str else "asr")

        return {
            "prompt": prompt_format.format(f"{prompt} {bias_str}"),
            "audio_path": audio_path,
            "text": bs.get_text().strip(),
            "keywords": entities,
            "id": utt_id,
        }

    return ds.map(load_sample)


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
            "prompt": prompt_format.format("Transcribe the audio clip into text."),
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


def stream_shuffle(ds, **kwargs):
    """Process the dataset."""
    streaming = kwargs.get("streaming", False)
    if streaming:
        ds = ds.to_iterable_dataset(num_shards=kwargs.get("num_shards", 1))
    num_egs = kwargs.get("num_egs", None)
    if num_egs is not None:
        ds = ds.take(num_egs)
    return ds


def bias_sampling(ds, **kwargs):
    """Apply bias sampling to the dataset."""
    rand_prompt = kwargs.pop("rand_prompt", False)

    kwargs = kwargs or {
        "bias_prob": 0.9,
        "hit_prob": 0.9,
        "max_piece_len": 1,
    }
    bias_sampler = PieceSampler(**kwargs)

    def proc_sample(sample):
        """Process a sample from the dataset."""
        context, text, keywords = bias_sampler.sample(sample["text"])
        if context:
            prompt = get_task_prompt(task="biasing", rand=rand_prompt)
            prompt = f"{prompt} {context}"
        else:
            prompt = get_task_prompt(task="asr", rand=rand_prompt)
        return {
            "prompt": prompt_format.format(prompt),
            "text": text,  # text is updated
            "keywords": keywords,
            "context": context,
        }

    ds = ds.map(proc_sample)
    return ds


def to_chat(text, chat=True, role="assistant"):
    """Convert text to conversation format."""
    if not chat:
        return text
    assert role in ["assistant", "user"], "Role must be either 'assistant' or 'user'."
    return [
        {
            "role": role,
            "content": text,
        }
    ]


def simulate_preference(ds, **kwargs):
    """simulate the preference  to the dataset."""
    error_range = kwargs.pop("error_range", (0.1, 0.25))
    num_rejections = kwargs.pop("num_rejections", 1)
    chat = kwargs.get("chat", False)
    if not isinstance(error_range, (tuple, list)):
        error_range = [float(error_range), float(error_range)]
    simulator = ErrorSimulator(**kwargs)

    def add_preference(sample, error_range):
        """Process a sample from the dataset."""
        text = sample["text"]
        rejections = [simulator.random_error(text, random.uniform(*error_range)) for _ in range(num_rejections)]
        return {
            "chosen": to_chat(text, chat),
            "rejected": [to_chat(x, chat) for x in rejections],
        }

    return ds.map(add_preference, fn_kwargs={"error_range": error_range})


def load_audio(ds):
    """Post process the dataset."""

    def read_audio(sample):
        """Read audio from the file."""
        audio, sr = sf_read(sample["audio_path"])
        return {"audio": audio, "sr": sr}

    ds = ds.map(read_audio)
    return ds


def filter_ds(ds, **kwargs):
    """Filter the dataset."""
    wer_file = kwargs.get("wer_file", None)
    if wer_file and bf.exists(wer_file):
        with bf.BlobFile(wer_file, "r") as f:
            df = pd.read_json(f, lines=True)
        if wer_range := kwargs.get("wer_range", None):
            df = df[(df["WER"] >= wer_range[0]) & (df["WER"] <= wer_range[1])]
        ids = df["id"].tolist()
        n_egs = len(ds)
        ds = ds.filter(lambda x: x["id"] in ids)
        print(f"Filter dataset: {n_egs} to {len(ds)}")
    return ds


def augment(ds, **kwargs):
    """Augment the dataset with additional information."""
    if filter_kwargs := kwargs.get("filter", {}):
        ds = filter_ds(ds, **filter_kwargs)
    if biasing_kwargs := kwargs.get("biasing", {}):
        ds = bias_sampling(ds, **biasing_kwargs)
    if pref_kwargs := kwargs.get("simu_preference", {}):
        ds = simulate_preference(ds, **pref_kwargs)
    if kwargs.get("load_audio", False):
        ds = load_audio(ds)
    return ds


def create_audio_dataset(dataset_name="openasr", **kwargs):
    """Create a dataset from the given split."""
    if dataset_name == "ls_bias":
        ds = ls_bias_dataset(**kwargs)
    elif dataset_name == "inhouse_entity":
        ds = entity_dataset(**kwargs)
    elif dataset_name == "openasr":
        ds = openasr_dataset(**kwargs)
        ds = augment(ds, **kwargs)
    elif dataset_name == "tsv":
        ds = tsv_dataset(**kwargs)
        ds = augment(ds, **kwargs)
    elif dataset_name == "chunk":
        ds = chunk_dataset(**kwargs)
        ds = augment(ds, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return ds


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
