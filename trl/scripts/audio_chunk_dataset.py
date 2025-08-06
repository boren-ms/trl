# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
import random
from tqdm import tqdm
import numpy as np
from math import ceil
import soundfile as sf
from cachetools import LRUCache
import blobfile as bf
from torch.utils.data import Dataset
import pandas as pd
from trl.trainer.utils import rank_print


def parse_data(data, data_type, **kwargs):
    if data_type.lower() == "audio":
        return sf.read(io.BytesIO(data))
    if data_type.lower() in ["info", "sft", "alignment"]:
        return json.loads(str(data, "utf-8"))
    if data_type.lower() == "feature":
        feat = np.frombuffer(data, dtype=np.float32)
        return feat.reshape(-1, kwargs.get("feat_dim", 80))
    return str(data, "utf-8")


class ChunkLoader:
    def __init__(self, chunk_path, chunk_type, count):
        self.chunk_path = chunk_path
        self.chunk_type = chunk_type
        self.count = count
        self._examples = None  # Will be loaded on demand
        self._unused = None

    def __repr__(self):
        return f"<ChunkLoader({self.chunk_path}, {self.chunk_type}, {self.count})>"

    def __len__(self):
        """Return the number of examples in the chunk."""
        return self.count

    def get(self, i):
        """Get the example at the specified index."""
        examples = self._get_examples()
        if i < 0 or i >= self.count:
            raise IndexError(f"Index {i} out of bounds for chunk with count {self.count}.")
        egs = examples[i]
        self._maybe_release(i)
        return egs

    def _maybe_release(self, i):
        """Release the loaded examples."""
        if self._unused is not None and i in self._unused:
            self._unused.remove(i)
        if not self._unused and self._examples is not None:
            rank_print(f"Releasing examples for chunk {self.chunk_path}.")
            del self._examples
            self._examples = None
            self._unused = None

    def _get_examples(self):
        """Load examples for the given index."""
        if self._examples is None:
            rank_print(f"Loading all examples for chunk {self.chunk_path}.")
            self._examples = load_data_from_chunk(self.chunk_path, self.chunk_type, self.count)
            self._unused = list(range(self.count))
        return self._examples


MAZ_LOADERS = 1000


class ChunkManager:
    """Manager for loading and managing chunks of data."""

    def __init__(self, maxsize=None):
        """Initialize the ChunkManager with a maximum size for the cache."""
        maxsize = maxsize or MAZ_LOADERS
        rank_print(f"Initializing ChunkManager with max {maxsize} ChunkLoaders.")
        self.chunk_loaders = LRUCache(maxsize=maxsize)

    def get(self, chunk_path, count, chunk_type=None):
        """Get the example at the specified index."""
        if chunk_path not in self.chunk_loaders:
            chunk_type = chunk_type or chunk_path.split(".")[-1]
            self.chunk_loaders[chunk_path] = ChunkLoader(chunk_path, chunk_type, count)
        return self.chunk_loaders[chunk_path]


def get_chunk_manager(maxsize=MAZ_LOADERS):
    """Return the global singleton ChunkManager."""
    global _chunk_manager_instance
    try:
        return _chunk_manager_instance
    except NameError:
        _chunk_manager_instance = ChunkManager(maxsize)
        return _chunk_manager_instance


def load_examples(chunk, types):
    examples = {}
    chunk_path = chunk.get("chunk_path", None)
    type_mapping = {"audio": "audio_path", "transcription": "trans_path"}
    count = chunk["count"]
    for chunk_type in types:
        chunk_type_key = type_mapping.get(chunk_type, chunk_type)
        chunk_type_path = chunk.get(chunk_type_key, chunk_path).rstrip("/")
        # rank_print(f"Loading {chunk_type} from {chunk_type_path} for chunk {chunk_name}")
        chunk_file = f"{chunk_type_path}/{chunk['name']}.{chunk_type}"
        assert bf.exists(chunk_file), f"Chunk file {chunk_file} does not exist."
        if chunk_type == "audio":
            examples["audio_chunk"] = [f"{chunk_file}:{count}:{i}" for i in range(count)]
        else:
            examples[chunk_type] = load_data_from_chunk(chunk_file, chunk_type, chunk["count"])

    df = pd.DataFrame(examples)
    return df.to_dict(orient="records")


def load_data_from_chunk(chunk_path: str, chunk_type: str, chunk_size: int):
    ENDIAN = "little"
    data_list = []
    with bf.BlobFile(chunk_path, "rb") as f:
        target_type = f.read(len(chunk_type.encode())).decode()
        if chunk_type.lower() != target_type.lower():
            raise ValueError(f"Target type is not expected in {chunk_path}, expected {chunk_type}, but got {target_type}")
        _ = int.from_bytes(f.read(4), byteorder=ENDIAN)
        for i in range(chunk_size):
            egs_i = int.from_bytes(f.read(4), byteorder=ENDIAN)
            if egs_i != i:
                raise ValueError(f"The example index is corrupted in {chunk_path}, expected {i}, but got {egs_i}")
            if target_type.lower() == "audios":
                parsed_data = []
                n_audios = int.from_bytes(f.read(4), byteorder=ENDIAN)
                for i in range(n_audios):
                    data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                    data = f.read(data_size)
                    parsed_data.append(parse_data(data, "audio"))
            else:
                data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                if target_type.lower() == "label":
                    data_size = int.from_bytes(f.read(2), byteorder=ENDIAN)
                data = f.read(data_size)
                parsed_data = parse_data(data, chunk_type)
            data_list.append(parsed_data)
    return data_list


def to_list(data):
    """Convert data to a list if it is not already."""
    if isinstance(data, (list, tuple)):
        return list(data)
    return [data]


def load_chunk_info(manifest_file, **kwargs):
    assert bf.exists(manifest_file), f"Chunk info file {manifest_file} does not exist."
    with bf.BlobFile(manifest_file, "r") as f:
        chunk_info = json.load(f)
    return [{**chunk, **kwargs} for chunk in chunk_info["fileInfo"]]


def load_specs(spec_files):
    """Load and return the specifications from the provided spec files."""
    specs = []
    for spec_file in to_list(spec_files):
        with bf.BlobFile(spec_file, "r") as f:
            spec_dict = json.load(f)
        specs += spec_dict["data_sources"]
    return specs


def load_chunks(specs, chunks_per_source=None):
    """Load and chunk dataset based on the provided data specification and chunk types."""
    if not isinstance(specs[0], dict):  # if specs is not list of dicts, assume list of files.
        specs = load_specs(specs)
    chunks = []
    rank_print(f"Loading chunks from {len(specs)} specs.")
    for spec in tqdm(specs, desc="Loading Specs"):
        chunks += load_chunk_info(**spec)[:chunks_per_source]
    rank_print(f"Loaded {len(chunks)} chunks.", f"Max chunks per source: {chunks_per_source}.")
    return chunks


def limit_chunks(chunks, max_egs=None, max_chunks=None):
    """Limit the number of chunks to max_chunks."""
    n_chunks = len(chunks)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
        rank_print(f"Limiting chunks {n_chunks} -> {len(chunks)} by max_chunks={max_chunks}.")
    if max_egs is not None:
        new_chunks = []
        total_egs = 0
        for chunk in chunks:
            if total_egs >= max_egs:
                break
            new_chunks.append(chunk)
            total_egs += chunk["count"]
        rank_print(f"Limiting chunks {n_chunks} -> {len(new_chunks)} by max_egs={max_egs}.")
        chunks = new_chunks
    return chunks


def generate_examples(specs, chunk_types=None, chunk_shuffle=True, max_chunks=None, max_egs=None):
    """Generate examples from the chunk dataset based on the specification files."""
    chunks_per_spec = ceil(max_chunks / len(specs)) if max_chunks else None
    chunks = load_chunks(specs, chunks_per_spec)
    chunks = random.shuffle(chunks) if chunk_shuffle else chunks
    chunks = limit_chunks(chunks, max_egs, max_chunks)
    types = to_list(chunk_types or ["audio", "transcription"])
    for chunk in tqdm(chunks, desc="Loading Chunks"):
        yield from load_examples(chunk, types)


def load_chunk_example(chunk_path):
    """Load a single example from the chunk file."""
    chunk_file, chunk_count, chunk_index = chunk_path.rsplit(":", 2)  # make sure rsplit.
    chunk_loader = get_chunk_manager().get(chunk_file, int(chunk_count))
    return chunk_loader.get(int(chunk_index))


# %%
if __name__ == "__main__":
    spec_file = [
        "/datablob1/users/ruchaofan/DataSpecs/mlang_s2/asr_person_filtered/asr_chunk_inhouse_en.json",
    ]
    # dataset = ChunkDataset(spec_file)
    # for i in range(0, 100, 10):  # Print every 10th sample, 50 samples in each chunk
    #     sample = dataset[i]
    #     rank_print(f"Sample {i}: {sample}")  # Output the sample data
    # pass
    for i, example in enumerate(generate_examples(spec_file, max_chunks=2)):
        rank_print(f"Example {i}: {example}")
        audio, fs = example["audio"]()
        rank_print(f"Audio shape: {audio.shape}, Sample rate: {fs}")
        if i > 52:
            break
            # %%
            break
# %%
