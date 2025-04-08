# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import backoff
import os
import io
import copy
import json
import logging
import math
import struct
import random
import threading
import torch
import torch.distributed as dist
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from numpy import random
import pandas as pd
import soundfile as sf
from torch.utils.data import IterableDataset
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

import warnings

from numpy import uint64, random

class KenslerPermutation:
    """
    A generator that returns a pseudo-random permutation of the numbers in [0, length - 1].
    The generator only uses O(1) internal state and retrieving the next number requires O(1) time on expectation
    (under the assumption that the internal hash-function is random...).
    This is in contrast to the often-used Fisher-Yates based algorithm that requires O(n) internal state,
    see https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

    The idea behind this class comes from Andrew Kensler's permute() function presented in
    Correlated Multi-Jittered Sampling
    Andrew Kensler
    Pixar Technical Memo 13-01
    see https://graphics.pixar.com/library/MultiJitteredSampling/paper.pdf

    For a good introduction to the ideas in this paper see
    https://andrew-helmer.github.io/permute/

    The hash function used in this class to support 64-bit integers was taken from
    https://github.com/camel-cdr/cauldron/blob/main/tools/random/permute/README.md
    """

    def __init__(self, length, seed=0):
        self.length = uint64(length)
        self.seed = random.default_rng(seed).integers(2**64, dtype=uint64)
        self.mask = KenslerPermutation.get_mask(self.length)

    def __iter__(self):
        return (self[i] for i in range(self.length))

    def __getitem__(self, i):
        return int(KenslerPermutation.permute(uint64(i), self.mask, self.seed, self.length))

    @staticmethod
    def get_mask(length):
        with warnings.catch_warnings():
            # ignore numpy overflow warnings as they are expected due to bit-wise operations
            warnings.simplefilter("ignore")
            mask = length - uint64(1)
            mask |= mask >> uint64(1)
            mask |= mask >> uint64(2)
            mask |= mask >> uint64(4)
            mask |= mask >> uint64(8)
            mask |= mask >> uint64(16)
            mask |= mask >> uint64(32)
            return mask

    @staticmethod
    def hash(index, mask, seed):
        # this hash function has been taken from
        # https://github.com/camel-cdr/cauldron/blob/main/tools/random/permute/README.md
        index ^= seed
        # splittable64
        index ^= (index & mask) >> uint64(30)
        index *= uint64(0xBF58476D1CE4E5B9)
        index ^= (index & mask) >> uint64(27)
        index *= uint64(0x94D049BB133111EB)
        index ^= (index & mask) >> uint64(31)
        index *= uint64(0xBF58476D1CE4E5B9)

        index ^= seed >> uint64(32)
        index &= mask
        index *= uint64(0xED5AD4BB)

        index ^= seed >> uint64(48)
        # hash16_xm3
        index ^= (index & mask) >> uint64(7)
        index *= uint64(0x2993)
        index ^= (index & mask) >> uint64(5)
        index *= uint64(0xE877)
        index ^= (index & mask) >> uint64(9)
        index *= uint64(0x0235)
        index ^= (index & mask) >> uint64(10)

        # From Andrew Kensler: "Correlated Multi-Jittered Sampling"
        index ^= seed
        index *= uint64(0xE170893D)
        index ^= seed >> uint64(16)
        index ^= (index & mask) >> uint64(4)
        index ^= seed >> uint64(8)
        index *= uint64(0x0929EB3F)
        index ^= seed >> uint64(23)
        index ^= (index & mask) >> uint64(1)
        index *= uint64(1) | seed >> uint64(27)
        index *= uint64(0x6935FA69)
        index ^= (index & mask) >> uint64(11)
        index *= uint64(0x74DCB303)
        index ^= (index & mask) >> uint64(2)
        index *= uint64(0x9E501CC3)
        index ^= (index & mask) >> uint64(2)
        index *= uint64(0xC860A3DF)
        index &= mask
        index ^= index >> uint64(5)

        return index

    @staticmethod
    def permute(index, mask, seed, length):
        with warnings.catch_warnings():
            # ignore numpy overflow warnings as they are expected due to bit-wise operations
            warnings.simplefilter("ignore")
            index = KenslerPermutation.hash(index, mask, seed)
            while index >= length:
                index = KenslerPermutation.hash(index, mask, seed)
            return (index + seed) % length
        
class DataParser:
    def __init__(self, feat_dim=80):
        self.feat_dim = feat_dim

    def parse_data(self, data, data_type):
        if data_type.lower() == "audio":
            parsed_data = self._parse_audio_data(data)
        elif data_type.lower() in ["info", "sft"]:
            parsed_data = self._parse_json_data(data)
        elif data_type.lower() == "feature":
            parsed_data = self._parse_feat_data(data)
        elif data_type.lower() == "label":
            parsed_data = self._parse_label_data(data)
        elif data_type.lower() == "alignment":
            parsed_data = self._parse_json_data(data)
        elif data_type.lower() == "single_qa":
            parsed_data = self._parse_eval_data(data)
        else:
            parsed_data = self._parse_string_data(data)
        return parsed_data

    def _parse_audio_data(self, data):
        byte_stream = io.BytesIO(data)
        return sf.read(byte_stream)

    def _parse_label_data(self, data):  # one dimension numpy array
        label_pairs = []
        for i in range(int(len(data) / 4)):
            label = struct.unpack_from("<h", data, i * 4)[0]
            repeat_num = struct.unpack_from("<h", data, i * 4 + 2)[0]
            label_pairs.append((label, repeat_num))

        labels = []
        for label_pair in label_pairs:
            labels.extend([label_pair[0]] * label_pair[1])

        return " ".join(list(map(str, labels)))

    def _parse_json_data(self, data):
        str_data = str(data, "utf-8")
        json_data = json.loads(str_data)
        return json_data

    def _parse_string_data(self, data):
        str_data = str(data, "utf-8")
        return str_data

    def _parse_feat_data(self, data):
        feat = np.frombuffer(data, dtype=np.float32)
        feat = feat.reshape(-1, self.feat_dim)
        return feat

    def _parse_eval_data(self, data):
        str_data = str(data, "utf-8")
        json_data = eval(str_data)
        return json_data


class DatasetSpec:
    """
    Data set specification

    This class parses a data set specification dictionary
    and hosts the ContainerClients used to connect to the Azure Blob Storage Accounts.

    The given dictionary is expected to conform with the following example:
    {
        "data_sources": [
            {
                "manifest_file": "/datablob/am_data/en_us_asr.json",
            },
            {
                "manifest_file": "/datablob/am_data/en_uk_asr.json",
            }
        ],
        "chunk_type": ["info", "audio", "transcription"]
    }

    WARNING: The chunk_index_range list is inclusive. It does not match Python's range.
    """

    def __init__(
        self,
        dataset_spec: dict,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        debug: bool = False,
        skip_blob_list: List = None,
    ):
        assert dataset_spec.keys() == {"data_sources"}
        self.__dict__.update(dataset_spec)

        if len(self.data_sources[0]["chunk_type"]) == 1 and self.data_sources[0]["chunk_type"][0] == "tsv":
            self.is_tsv = True
        else:
            self.is_tsv = False

        self.debug = debug
        self.skip_blob_list = skip_blob_list

        self.chunk_sizes = []
        self.total_chunks = 0
        self.total_durations = 0
        self.accum_chunks = []
        for data_source in self.data_sources:
            chunk_list = self.load_manifest_file(data_source)
            if self.is_tsv:
                n_examples = [len(eval(m)) for m in chunk_list["messages"]]
                self.total_durations += sum(n_examples)
                assert use_num_examples_as_durations, "Please use examples as weight for tsv data!"
                mean_examples = sum(n_examples) / len(n_examples)
                in_tsv_chunk_size = int(tsv_chunk_size // mean_examples)

                # split on tsv files into chunks
                total_chunks = math.ceil(len(chunk_list) / in_tsv_chunk_size)
                data_source["total_chunks"] = total_chunks
                self.total_chunks += total_chunks
                last_chunk_size = len(chunk_list) - in_tsv_chunk_size * (total_chunks - 1)
                chunk_sizes = [in_tsv_chunk_size] * (total_chunks - 1) + [last_chunk_size]
                self.chunk_sizes.extend(chunk_sizes)

                data_source["local_chunk_sizes"] = chunk_sizes
                data_source["chunk_tsv_offsets"] = self.get_tsv_chunk_offset(data_source["manifest_file"], chunk_sizes)
            else:
                data_source["total_chunks"] = len(chunk_list)
                self.total_chunks += data_source["total_chunks"]
                for chunk in chunk_list:
                    self.chunk_sizes.append(int(chunk["count"]))
                    example_count = int(chunk.get("example_count", chunk["count"]))
                    self.total_durations += float(chunk["duration"]) if not use_num_examples_as_durations else example_count
            self.accum_chunks.append(self.total_chunks)
        assert len(self.chunk_sizes) == self.total_chunks

        self.parser = DataParser()

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def load_manifest_file(self, data_source, n_chunks=-1):
        full_path = data_source["manifest_file"]
        if self.is_tsv:
            col_names = ["id", "wav_paths", "messages"]
            if n_chunks == -1:
                return pd.read_csv(full_path, names=col_names, sep="\t")
            else:
                nrows = data_source["local_chunk_sizes"][n_chunks]
                # chunk_data =  pd.read_csv(full_path, names=col_names, skiprows=n_chunks * data_source["in_tsv_chunk_size"], nrows=nrows, sep="\t")
                # pandas reading may cause high cpu memory usage, we save the file offset for each chunk
                offset = data_source["chunk_tsv_offsets"][n_chunks]
                chunk_data = {"id": [], "wav_paths": [], "messages": []}

                with open(full_path, "rb") as file:
                    file.seek(offset)
                    return pd.read_csv(file, names=col_names, nrows=nrows, sep="\t")
                return chunk_data
        else:
            with open(full_path, "r", encoding="utf-8") as f:
                return json.load(f)["fileInfo"]

    def get_tsv_chunk_offset(self, tsv_file, chunk_sizes):
        offsets = []

        with open(tsv_file, "rb") as file:
            offset = 0
            for i in range(len(chunk_sizes)):
                offsets.append(offset)
                offset += sum(len(file.readline()) for _ in range(chunk_sizes[i]))
            assert file.readline() == b""
        return offsets

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def load_chunk_with_chunkname(self, blob_name: str, chunk_type: str, chunk_size: int):
        if self.skip_blob_list is not None and blob_name in self.skip_blob_list:
            logging.error(f"chunk {blob_name} in missing file/skip list. Skip and return None.")
            return [None for _ in range(chunk_size)]
        ENDIAN = "little"
        example_list = []
        with open(blob_name, "rb") as f:
            target_type = f.read(len(chunk_type.encode())).decode()
            if chunk_type.lower() != target_type.lower():
                raise ValueError(f"Target type is not expected in {blob_name}, expected {chunk_type}, but got {target_type}")
            version_number = int.from_bytes(f.read(4), byteorder=ENDIAN)

            for i in range(chunk_size):
                example_index = int.from_bytes(f.read(4), byteorder=ENDIAN)
                if example_index != i:
                    raise ValueError(f"The example index is corrupted in {blob_name}, expected {i}, but got {example_index}")
                if target_type.lower() == "audios":
                    parsed_data = []
                    n_audios = int.from_bytes(f.read(4), byteorder=ENDIAN)
                    for i in range(n_audios):
                        data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                        data = f.read(data_size)
                        parsed_data.append(self.parser.parse_data(data, "audio"))
                else:
                    data_size = int.from_bytes(f.read(4), byteorder=ENDIAN)
                    if target_type.lower() == "label":
                        data_size = int.from_bytes(f.read(2), byteorder=ENDIAN)
                    data = f.read(data_size)
                    parsed_data = self.parser.parse_data(data, chunk_type)
                example_list.append(parsed_data)

        return example_list

    def load_chunk(self, global_chunk_index: int):
        """
        download and return chunk data in form List[], each item in the list represents one sample

        Args:
            global_chunk_index (int): _description_

        Returns:
            _type_: _description_
        """

        assert 0 <= global_chunk_index < self.total_chunks
        # determine where the chunk should be fetched from
        data_source_index = np.searchsorted(self.accum_chunks, global_chunk_index, side="right")
        data_source = self.data_sources[data_source_index]
        local_chunk_index = global_chunk_index - (self.accum_chunks[data_source_index - 1] if data_source_index > 0 else 0)
        if self.is_tsv:
            chunk_list = self.load_manifest_file(data_source, n_chunks=local_chunk_index)
            all_chunk_data = []
            for i in range(len(chunk_list["wav_paths"])):
                chunk_data = {}
                wav_paths = eval(chunk_list["wav_paths"][i])
                chunk_data["audios"] = [safe_read(wav_path) for wav_path in wav_paths]
                chunk_data["sft"] = eval(chunk_list["messages"][i])
                all_chunk_data.append(chunk_data)
            chunk_type = ["audios", "sft"]
        else:
            chunk_list = self.load_manifest_file(data_source)
            chunk_name = chunk_list[local_chunk_index]["name"]
            chunk_type = data_source["chunk_type"]
            chunk_size = chunk_list[local_chunk_index]["count"]
            all_chunk_data = [{} for _ in range(chunk_size)]
            for extension in chunk_type:
                additional_path = data_source.get("additional_path")
                if additional_path is not None and extension not in [
                    "audio",
                    "feature",
                    "info",
                    "transcription",
                    "audios",
                    "sft",
                ]:
                    assert extension in additional_path, f"pls provide path for {extension}"
                    path = additional_path[extension]
                elif extension in ["transcription", "sft"] and data_source.get("trans_path", None) is not None:
                    path = data_source["trans_path"]
                else:
                    path = data_source["chunk_path"]

                blob_name = os.path.join(path, chunk_name + "." + extension)
                try:
                    chunk_data = self.load_chunk_with_chunkname(blob_name, extension, chunk_size)
                except Exception as e:
                    logger.info(f"decode chunk failed data source {data_source['manifest_file']}, blob name {blob_name}, return empty list" + str(e))
                    chunk_data = [None for _ in range(chunk_size)]

                for i in range(chunk_size):
                    all_chunk_data[i][extension] = chunk_data[i]

        # for i in range(len(all_chunk_data)):
        #     if any(all_chunk_data[i][key] is None for key in chunk_type):
        #         all_chunk_data[i] = {}
        #     else:
        #         all_chunk_data[i]["language"] = data_source["language"]
        for chunk in all_chunk_data:
            chunk["language"] = data_source["language"]

        return all_chunk_data


class BlobChunkIterableDataset(IterableDataset):
    """
    An IterableDataset for efficient large-scale data loading

    Args:
        dataset_spec (dict):
            Dataset specification dictionary.
            See class `DatasetSpec` above for more information.
        decode_chunk_fn (Callable):
            A function used to decode the binary representation of a chunk.
            Typically, you would want to use one of the functions
            `decode_tsv_chunk` or `decode_binary_chunk` defined above.
        num_chunks_for_shuffling (int):
            Number of chunks that should be used for shuffling.
            Each worker is going to prefetch this number of chunks,
            shuffles the examples from all of these chunks together,
            and serves them out.
        decode_example_fn (Callable):
            An optional function used to decode an example.
            If None, no function is applied an example.
            Default: None
        num_replicas (int):
            Total number of GPUs / data set instances in distributed training.
            If None, use torch.distributed to determine num_replicas.
            Default: None
        rank (int):
            Rank of this GPU / data set instance in distributed training.
            If None, use torch.distributed to determine rank.
            Default: None
        seed (int):
            Seed for shuffling.
        multi_threaded_download (bool):
            If True, use threads to download chunks, one thread per chunk.
            Default: False
        dummy_data_mode (bool):
            Used for debugging and unit tests.
            If True, the data set does not connect to an Azure storage account
            but instead just returns integers.
            Default: False
        debug (bool):
            If True, append debug info when decoding examples.
            Default: False
        skip_blob_list (List[str]): A list of blob file name. Will skip reading these blob files.
    """

    def __init__(
        self,
        dataset_spec: dict,
        num_chunks_for_shuffling: int,
        num_replicas: int = None,
        rank: int = None,
        seed: int = 0,
        multi_threaded_download: bool = False,
        dummy_data_mode: bool = False,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        debug: bool = False,
        skip_blob_list=None,
    ):
        super().__init__()
        self.dataset_spec = DatasetSpec(dataset_spec, use_num_examples_as_durations, tsv_chunk_size, debug, skip_blob_list=skip_blob_list)
        self.num_chunks_for_shuffling = num_chunks_for_shuffling

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1))

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.multi_threaded_download = multi_threaded_download
        self.dummy_data_mode = dummy_data_mode

        self.epoch = 0
        self.step_in_epoch = 0

    def set_epoch(self, epoch: int):
        """
        Set epoch

        The next time you iterate over the data set after calling this function,
        it will start at the given epoch.

        Args:
            epoch (int): epoch from which data loading should start
        """
        self.epoch = epoch
        self.step_in_epoch = 0

    def resume_from_checkpoint(self, epoch: int, step_in_epoch: int):
        """
        Resume data loading from a checkpoint

        The next time you iterate over the data set after calling this function,
        it will resume at the specified epoch and step_in_epoch.

        WARNING:
        When resuming a job that used a specific value for num_workers in the data loader,
        you must make sure the resuming job uses the same value both in the data loader
        and in the constructor of this class.

        Args:
            epoch (int): epoch from which data loading should resume
            step_in_epoch (int): step in epoch from which data loading should resume
        """
        self.epoch = epoch
        self.step_in_epoch = step_in_epoch

    def __iter__(self):
        assert self.rank is not None and self.num_replicas is not None, "Distributed config not set."
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # assert self.num_workers == 0, f"unexpected number of workers {self.num_workers} when worker_info is None"
            worker_step_in_epoch = self.step_in_epoch
            num_replicas = self.num_replicas
            rank = self.rank
        else:
            # assert self.num_workers == worker_info.num_workers, f"unexpected number of workers self.num_workers={self.num_workers} worker_info.num_workers={worker_info.num_workers}"
            self.num_workers = worker_info.num_workers

            # In order to correctly resume data loading from a specific step,
            # we have to make sure to correctly handle the case the the PyTorch data loader
            # uses multiple worker processes.
            #
            # Let's say we have 4 worker processes.
            # The PyTorch data loader consumes examples from these workers in a round-robin fashion.
            # Let's say we want to resume at the global step 6,
            # then worker 2 (= 6 % 4) has to supply the next example.
            # However, the PyTorch data loader will always get the first example from worker 0.
            # To fix this, we rotate the data worker ids so that, in this example, worker 2 becomes worker 0.
            # Additionally, we have to skip an additional example on some of the worker,
            # namely worker 0 and 1 in our example.
            #
            # The logic below achieves exactly that.
            num_workers = max(self.num_workers, 1)
            worker_id_offset = self.step_in_epoch % num_workers

            worker_id = (worker_info.id + worker_id_offset) % num_workers

            num_replicas = self.num_replicas * num_workers
            rank = self.rank * num_workers + worker_id

            worker_step_in_epoch = self.step_in_epoch // num_workers
            if worker_id < worker_id_offset:
                worker_step_in_epoch += 1

        it = _BlobChunkIterableDatasetIterator(
            self.dataset_spec,
            self.num_chunks_for_shuffling,
            self.epoch,
            worker_step_in_epoch,
            num_replicas,
            rank,
            self.seed,
            self.multi_threaded_download,
            self.dummy_data_mode,
        )

        # in the next epoch, we want to start at the beginning
        self.step_in_epoch = 0

        return it


def safe_read(wav_path):
    """Read wav file, return None if failed"""
    try:
        return sf.read(wav_path)
    except Exception as e:
        logger.info(f"read wav failed file name {wav_path}, return None " + str(e))
        return None


class _ChunkIndexPermutation:
    """
    This helper class contains the logic necessary to:
        - shuffle chunks
        - distribute the chunks across GPU and / or data loader workers
        - translate between (local) worker chunk indices and global chunk indices
        - make sure that all workers receive the same number of chunks
    """

    def __init__(self, total_chunks, num_replicas, rank, seed, epoch):
        self.total_chunks = total_chunks
        self.num_replicas = num_replicas
        self.rank = rank

        self.chunk_permutation = KenslerPermutation(length=self.total_chunks, seed=seed + epoch)

    def __getitem__(self, worker_chunk_index):
        global_chunk_index = self.rank + worker_chunk_index * self.num_replicas
        return self.chunk_permutation[global_chunk_index]


class _BlobChunkIterableDatasetIterator:
    def __init__(
        self,
        dataset_spec: DatasetSpec,
        num_chunks_for_shuffling: int,
        epoch: int,
        worker_step_in_epoch: int,
        num_replicas: int,
        rank: int,
        seed: int,
        multi_threaded_download: bool,
        dummy_data_mode: bool,
    ):
        self.dataset_spec = dataset_spec
        self.num_chunks_for_shuffling = num_chunks_for_shuffling
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = epoch
        self.worker_step_in_epoch = worker_step_in_epoch
        self.seed = seed
        self.multi_threaded_download = multi_threaded_download
        self.dummy_data_mode = dummy_data_mode

        self.resuming_from_checkpoint = self.worker_step_in_epoch != 0

        self.total_chunks = self.dataset_spec.total_chunks

        # TODO (Yi-Ling): Currently, this class requires that the number of chunks is greater or equal to the global number of replicas (#gpus * #workers_per_gpu). This might be a problem when training on a small data set with many GPUs. To address this, we could add a feature to "virtually duplicate" all chunks. Effectively, this would mean that we do multiple sweeps through the data in a single epoch. To implement this, we would have to modify `_ChunkIndexPermutation` and some of the logic in this class.
        assert self.total_chunks >= self.num_replicas, f"too few chunks ({self.total_chunks}) in data set for number of replicas ({self.num_replicas})"

        self.chunk_index_permutation = _ChunkIndexPermutation(self.total_chunks, self.num_replicas, self.rank, self.seed, self.epoch)

        worker_permuted_chunk_sizes = [self.dataset_spec.chunk_sizes[self.chunk_index_permutation.chunk_permutation[i]] for i in range(self.rank, self.total_chunks, self.num_replicas)]
        self.worker_accum_permuted_chunk_sizes = np.cumsum(worker_permuted_chunk_sizes)
        self.worker_total_num_examples = self.worker_accum_permuted_chunk_sizes[-1]
        self.worker_accum_buffered_chunk_sizes = [self.worker_accum_permuted_chunk_sizes[-1]] if len(worker_permuted_chunk_sizes) < self.num_chunks_for_shuffling else self.worker_accum_permuted_chunk_sizes[self.num_chunks_for_shuffling - 1 :: self.num_chunks_for_shuffling]
        if len(worker_permuted_chunk_sizes) > self.num_chunks_for_shuffling and len(worker_permuted_chunk_sizes) % self.num_chunks_for_shuffling != 0:
            assert self.worker_accum_buffered_chunk_sizes[-1] < self.worker_total_num_examples
            self.worker_accum_buffered_chunk_sizes = np.concatenate((self.worker_accum_buffered_chunk_sizes, [self.worker_total_num_examples]))

        self.example_buffer = []
        self.example_buffer_permutation = None

    def __next__(self):
        worker_step_this_sweep = self.worker_step_in_epoch % self.worker_total_num_examples
        search_index = np.searchsorted(self.worker_accum_buffered_chunk_sizes, worker_step_this_sweep)
        if worker_step_this_sweep == 0 or self.worker_accum_buffered_chunk_sizes[search_index] == worker_step_this_sweep:
            example_index = 0
        else:
            example_index = worker_step_this_sweep if search_index == 0 else worker_step_this_sweep - self.worker_accum_buffered_chunk_sizes[search_index - 1]

        data_exhausted = False
        if self.resuming_from_checkpoint or example_index == 0:
            data_exhausted = self._replenish_example_buffer()
            self.resuming_from_checkpoint = False

        if not data_exhausted and len(self.example_buffer) == 0:
            logging.warning(
                "Empty buffer after replenishment. Advance to the next chunks. \
                            Dangerous operations if you uses multiple iter-type data, when only one data is moved to next."
            )
            self.worker_step_in_epoch += block_size
            return next(self)

        if example_index >= len(self.example_buffer):
            logger.debug(f"Data set exhausted at rank {dist.get_rank()}")
            raise StopIteration  # data set exhausted

        shuffled_example_index = self.example_buffer_permutation[example_index]
        assert shuffled_example_index < len(self.example_buffer)

        self.worker_step_in_epoch += 1
        example = self.example_buffer[shuffled_example_index]

        return example

    def __iter__(self):
        return self

    def _replenish_example_buffer(self):
        worker_step_this_sweep = self.worker_step_in_epoch % self.worker_total_num_examples
        search_index = np.searchsorted(self.worker_accum_permuted_chunk_sizes, worker_step_this_sweep)
        assert self.resuming_from_checkpoint or worker_step_this_sweep == 0 or self.worker_accum_permuted_chunk_sizes[search_index] == worker_step_this_sweep
        if self.resuming_from_checkpoint:
            idx = np.searchsorted(self.worker_accum_buffered_chunk_sizes, worker_step_this_sweep)
            if self.worker_accum_buffered_chunk_sizes[idx] == worker_step_this_sweep:
                search_index = (idx + 1) * self.num_chunks_for_shuffling - 1
            else:
                search_index = idx * self.num_chunks_for_shuffling - 1
        worker_chunk_index_starts_from = 0 if worker_step_this_sweep == 0 else search_index + 1
        this_num_chunks_for_shuffling = min(
            self.num_chunks_for_shuffling,
            len(self.worker_accum_permuted_chunk_sizes) - worker_chunk_index_starts_from,
        )

        self.example_buffer = []
        threads = []
        thread_results = [None] * this_num_chunks_for_shuffling
        for i in range(this_num_chunks_for_shuffling):
            worker_chunk_index = worker_chunk_index_starts_from + i
            global_chunk_index = self.chunk_index_permutation[worker_chunk_index]

            if i == 0:
                first_global_chunk_index = global_chunk_index  # used to compute seed for example buffer permutation

            if global_chunk_index is None:  # chunks exhausted:
                if i == 0:  # end of data set
                    return True
                else:  # some remaining data to be processed
                    break

            if self.multi_threaded_download:
                t = threading.Thread(target=self._read_chunk, args=(global_chunk_index, i, thread_results))
                t.start()
                threads.append(t)
            else:
                self._read_chunk(global_chunk_index, i, thread_results)

        # wait for threads to complete
        if self.multi_threaded_download:
            for t in threads:
                t.join()

        # collect results
        for result in thread_results:
            if result is not None:
                self.example_buffer += result

        del thread_results
        # compute permutation used to shuffle examples in example buffer
        seed = self.seed + (self.worker_step_in_epoch // self.worker_total_num_examples) * self.total_chunks + first_global_chunk_index
        self.example_buffer_permutation = KenslerPermutation(len(self.example_buffer), seed=seed)
        return False  # there are still remaining data

    def _read_chunk(self, chunk_index: int, thread_id: int, thread_results: List):
        if self.dummy_data_mode:
            examples = list(
                range(
                    self.worker_accum_permuted_chunk_sizes[chunk_index],
                    self.worker_accum_permuted_chunk_sizes[chunk_index + 1],
                )
            )
        else:
            examples = self.dataset_spec.load_chunk(chunk_index)
        thread_results[thread_id] = examples


class ChunkSpeechTextSeq2SeqDataset(IterableDataset):
    """
    An IterableDataset for efficient large-scale data loading of image-text pairs for seq-to-seq modeling
    supporting multiple text for each image (e.g. caption, OD, face).

    Args:
        dataset_spec (dict):
            Dataset specification dictionary.
            See example below.
        num_chunks_for_shuffling (int):
            Number of chunks that should be used for shuffling.
            Each worker is going to prefetch this number of chunks,
            shuffles the examples from all of these chunks together,
            and serves them out.
        transforms (Callable):
            An optional function used transform the images.
            Default: None
        tokenize (Callable):
            An optional function used transform the text.
            Default: None
        context_length (int):
            The context length to use; We follow CLIP models to use 77 as the default context length.
            Default: 77
        num_replicas (int):
            Total number of GPUs / data set instances in distributed training.
            If None, use torch.distributed to determine num_replicas.
            Default: None
        rank (int):
            Rank of this GPU / data set instance in distributed training.
            If None, use torch.distributed to determine rank.
            Default: None
        seed (int):
            Seed for shuffling.
        multi_threaded_download (bool):
            If True, use threads to download chunks, one thread per chunk.
            Default: False
        debug (bool):
            If True, append debug info when decoding examples.
            Default: False
        skip_blob_list (List[str]): A list of blob file name. Will skip reading these blob files.
        skip_missing_data (bool): whether skip data if missing or wrongly decoded.

    Example dataset_spec:
    {
        "total_chunks": 100,
        "examples_per_chunk": 1000,
        "data_sources": [
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
            },
            {
                "total_chunks": 50,
                "storage_account_url": "https://dataloading.blob.core.windows.net/",
                "container_name": "data",
                "SAS": "SECRET",
            },
        ]
    }
    """

    def __init__(
        self,
        dataset_spec: dict,
        num_chunks_for_shuffling: int,
        transforms: Callable = None,
        tokenize: Callable = None,
        context_length: int = 77,
        num_replicas: int = None,
        rank: int = None,
        drop_last: bool = False,  # no use
        seed: int = 0,
        multi_threaded_download: bool = False,
        use_num_examples_as_durations: bool = False,
        tsv_chunk_size: int = 256,
        debug: bool = False,
        skip_blob_file_path=None,
        skip_missing_data=False,
        tokenize_padding="max_length",
    ):
        self.transforms = transforms
        self.tokenize = tokenize
        if tokenize_padding is None:
            tokenize_padding = "do_not_pad"
        self.tokenize_padding = tokenize_padding
        self.context_length = context_length
        self.skip_missing_data = skip_missing_data
        if skip_blob_file_path is not None and os.path.exists(skip_blob_file_path):
            skip_blob_list = []
            with open(skip_blob_file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                skip_blob_list.append(line.strip())
            logging.info("all skip blob list {}".format(skip_blob_list))
        else:
            skip_blob_list = None

        self.use_num_examples_as_durations = use_num_examples_as_durations
        self.tsv_chunk_size = tsv_chunk_size
        self.debug = debug

        self.audio_dataset = BlobChunkIterableDataset(
            dataset_spec=dataset_spec,
            num_chunks_for_shuffling=num_chunks_for_shuffling,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            multi_threaded_download=multi_threaded_download,
            use_num_examples_as_durations=use_num_examples_as_durations,
            tsv_chunk_size=tsv_chunk_size,
            debug=debug,
            skip_blob_list=skip_blob_list,
        )

        self.num_samples = sum(self.audio_dataset.dataset_spec.chunk_sizes)
        self.total_durations = self.audio_dataset.dataset_spec.total_durations
        if use_num_examples_as_durations:
            logger.info("Total Training Audio Text Pairs: {}, Total Training Examples {}".format(self.num_samples, self.total_durations))
        else:
            logger.info("Total Training Audio Text Pairs: {}, Total Duration: {} hours".format(self.num_samples, self.total_durations / 3600))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """
        Set epoch

        The next time you iterate over the data set after calling this function,
        it will start at the given epoch.

        Args:
            epoch (int): epoch from which data loading should start
        """
        random.seed(epoch)
        self.audio_dataset.set_epoch(epoch)

    def set_state(self, state: Dict[str, Any]):
        self.resume_from_checkpoint(state["epoch"], state["generator_state"])

    def set_rank_and_world_size(self, rank: Optional[int], world_size: Optional[int]) -> None:
        if torch.distributed.is_initialized():
            rank = rank if rank is not None else torch.distributed.get_rank()
            world_size = world_size if world_size is not None else torch.distributed.get_world_size()
        else:
            rank, world_size = 0, 1

        self.audio_dataset.rank = rank
        self.audio_dataset.num_replicas = world_size

    def resume_from_checkpoint(self, epoch: int, step_in_epoch: int):
        """
        Resume data loading from a checkpoint

        The next time you iterate over the data set after calling this function,
        it will resume at the specified epoch and step_in_epoch.

        WARNING:
        When resuming a job that used a specific value for num_workers in the data loader,
        you must make sure the resuming job uses the same value both in the data loader
        and in the constructor of this class.

        Args:
            epoch (int): epoch from which data loading should resume
            step_in_epoch (int): step in epoch from which data loading should resume
        """

        logger.info(f"Resume data loader from checkpoint: epoch({epoch}), step_in_epoch({step_in_epoch})")
        self.audio_dataset.resume_from_checkpoint(epoch, step_in_epoch)

    def __iter__(self):
        raise NotImplementedError("This class is not meant to be used directly. Use one of the subclasses.")
