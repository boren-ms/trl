# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
import pandas as pd


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


def load_examples(chunk, types):
    examples = {}
    chunk_path = chunk.get("chunk_path", None)
    type_mapping = {
        "audio": "audio_path",
        "transcription": "trans_path",
    }
    chunk_name = chunk.get("name", None)
    for chunk_type in types:
        chunk_type_key = type_mapping.get(chunk_type, chunk_type)
        chunk_type_path = chunk.get(chunk_type_key, chunk_path)
        print(f"Loading {chunk_type} from {chunk_type_path} for chunk {chunk_name}")
        chunk_file = Path(chunk_type_path) / f"{chunk_name}.{chunk_type}"
        assert chunk_file.exists(), f"Chunk file {chunk_file} does not exist."
        examples[chunk_type] = load_data_from_chunk(chunk_file, chunk_type, chunk["count"])

    df = pd.DataFrame(examples)
    return df.to_dict(orient="records")


def load_data_from_chunk(chunk_path: str, chunk_type: str, chunk_size: int):
    ENDIAN = "little"
    data_list = []
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
            data_list.append(parsed_data)
    return data_list


def load_chunks(spec_file):
    """Load and chunk dataset based on the provided data specification and chunk types."""
    with open(spec_file, "r", encoding="utf-8") as f:
        spec_dict = json.load(f)
    chunks = []
    for data_source in spec_dict["data_sources"]:
        chunks += load_chunk_info(**data_source)
    return chunks


class ChunkDataset(Dataset):
    """Dataset class for loading and managing audio chunks based on a specification file."""

    def __init__(self, spec_file, chunk_types=None):
        self.chunks = load_chunks(spec_file)
        self.types = chunk_types or ["audio", "transcription"]

        print(f"Loaded dataset with {len(self.chunks)} chunks for types {self.types}.")
        self.samples = []  # (chunk_idx, chunk_shift)
        for idx, chunk in enumerate(self.chunks):
            for shift in range(chunk["count"]):
                self.samples.append((idx, shift))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_idx, shift = self.samples[idx]
        chunk = self.chunks[chunk_idx]
        assert 0 <= shift < chunk["count"], f"Shift {shift} out of bounds for chunk {chunk_idx} with count {chunk['count']}."
        examples = chunk.get("examples", None)
        if examples is None:
            print(f"Loading examples for chunk {chunk_idx} for shift {shift}.")
            # TODO: consider to delete the examples after loading
            examples = load_examples(chunk, self.types)
            chunk["examples"] = examples

        return examples[shift]


# %%
if __name__ == "__main__":
    # spec_file = "/datablob1/users/ruchaofan/DataSpecs/mlang_s2/asr_person_filtered/asr_chunk_inhouse_en.json"
    # chunk_types = ["audio", "transcription"]
    # dataset = ChunkDataset(spec_file, chunk_types)
    # for i in range(0, 100, 10):  # Print every 10th sample, 50 samples in each chunk
    #     sample = dataset[i]
    #     print(f"Sample {i}: {sample}")  # Output the sample data
    pass

# %%
