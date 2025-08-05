# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import Dataset


def parse_data(data, data_type, **kwargs):
    if data_type.lower() == "audio":
        return sf.read(io.BytesIO(data))
    if data_type.lower() in ["info", "sft", "alignment"]:
        return json.loads(str(data, "utf-8"))
    if data_type.lower() == "feature":
        feat = np.frombuffer(data, dtype=np.float32)
        return feat.reshape(-1, kwargs.get("feat_dim", 80))
    return str(data, "utf-8")


def load_chunk_info(manifest_file, **kwargs):
    assert Path(manifest_file).exists(), f"Chunk info file {manifest_file} does not exist."
    with open(manifest_file, "r", encoding="utf-8") as f:
        chunk_info = json.load(f)
    return [{**chunk, **kwargs} for chunk in chunk_info["fileInfo"]]


def example_count(chunks):
    return sum(chunk["count"] for chunk in chunks)


def load_examples(chunk, types):
    examples = {}
    chunk_path = chunk.get("chunk_path", None)

    for chunk_type in types:
        chunk_type_path = Path(chunk.get("chunk_type", chunk_path))
        assert chunk_type_path is not None, f"Chunk type path for {chunk_type} is not provided in {chunk}"
        chunk_file = chunk_type_path / chunk["name"] + "." + chunk_type
        assert chunk_file.exists(), f"Chunk file {chunk_file} does not exist."
        examples[chunk_type] = load_chunk_with_chunkname(chunk_file, chunk_type, chunk["count"])
    return examples


def load_chunk_with_chunkname(chunk_path: str, chunk_type: str, chunk_size: int):
    ENDIAN = "little"
    examples = []
    with open(chunk_path, "rb") as f:
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
            examples.append(parsed_data)
    return examples


def chunk_dataset(spec_file, chunk_types=None):
    """Load and chunk dataset based on the provided data specification and chunk types."""
    with open(spec_file, "r", encoding="utf-8") as f:
        spec_dict = json.load(f)
    chunk_types = chunk_types or spec_dict.get("chunk_type", ["audio", "transcription"])
    if isinstance(chunk_types, str):
        chunk_types = [chunk_types]
    chunks = []
    for data_source in spec_dict["data_sources"]:
        chunks += load_chunk_info(**data_source)
    # examples = [load_examples(chunk, chunk_types) for chunk in chunks]
    return Dataset.from_list(chunks)


# %%
spec_file = "/datablob1/users/ruchaofan/DataSpecs/mlang_s2/asr_person_filtered/asr_chunk_inhouse_en.json"
chunk_types = ["audio", "transcription"]

ds = chunk_dataset(spec_file, chunk_types)
print(f"Loaded dataset with {len(ds)} chunks.")

# %%
