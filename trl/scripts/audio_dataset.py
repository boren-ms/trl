# %%
import os
import ast
import urllib
import random
import blobfile as bf
import pandas as pd
import string
from functools import partial
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from bs4 import BeautifulSoup
from trl.scripts.error_simu import ErrorSimulator
from trl.scripts.biasing import PieceSampler, tag_pieces, text_norm as biasing_text_norm
from trl.scripts.audio_prompts import get_task_prompt
from trl.scripts.audio_metrics import text_norm
from trl.scripts.shared_utils import get_config_path
from trl.scripts.chunk_dataset import generate_examples, get_chunk_manager, to_list
from trl.data_utils import sf_read
from trl.trainer.utils import rank_print

prompt_format = "<|user|><|audio_1|>{}<|end|><|assistant|>"


def read_words(file_path, num=None, tn_name=None):
    """Read the top N lines from a file."""
    words = []
    tn_name = tn_name or "identity"
    with bf.BlobFile(file_path, "r") as f:
        for i, line in enumerate(f):
            if num is not None and i >= num:
                break
            word = line.split()[0]
            words.append(text_norm(word, tn_name))
    return words


def prefix_match(str1, str2, nd=2):
    n = min(len(str1), len(str2))
    m = max(len(str1), len(str2))
    if m - n > nd:
        return False
    return str1[:n] == str2[:n]


def has_digit(s):
    return any(c.isdigit() for c in s)


def find_rare(srcs, tgts, nd=2):
    srcs = set(srcs) - set(tgts)
    lefts = []
    for src in srcs:
        if any(prefix_match(src, tgt, nd=nd) for tgt in tgts):
            continue
        if has_digit(src):
            continue
        lefts.append(src)
    return lefts


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
        words = tag_pieces(words, tag=tag, specified=gt_words, norm=biasing_text_norm)
        return {
            "prompt": prompt_format.format(f"{prompt} {bias_str}"),
            "audio_path": audio_path,
            "text": " ".join(words),
            "keywords": gt_words,
            "id": example.get("id", Path(audio_path).stem),
        }

    ds = ds.map(load_sample, num_proc=kwargs.get("num_proc", 1))
    return ds


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

    return ds.map(load_sample, num_proc=kwargs.get("num_proc", 1))


def load_tsv(tsv_file, **kwargs):
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
    ds = ds.map(lambda x: {"dir": dir_path}, num_proc=kwargs.get("num_proc", 1))
    return ds


def tsv_dataset(tsv_paths, **kwargs):
    """Create a dataset from the given split."""
    if isinstance(tsv_paths, (list, tuple)):
        ds = concatenate_datasets([load_tsv(tsv_path, **kwargs) for tsv_path in tsv_paths])
    else:
        ds = load_tsv(tsv_paths, **kwargs)

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

    ds = ds.map(load_sample, num_proc=kwargs.get("num_proc", 1))
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

    ds = ds.map(proc_sample, num_proc=kwargs.get("num_proc", 1))
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


def format_preference(ds, **kwargs):
    """Format the preference for the dataset."""
    chosen_key = kwargs.get("chosen_key", "chosen")
    rejected_key = kwargs.get("rejected_key", "rejected")
    prompt_key = kwargs.get("prompt_key", "prompt")

    def format_sample(sample):
        """Format a single sample."""
        return {
            "prompt": sample.get(prompt_key, None),
            "chosen": sample.get(chosen_key, None),
            "rejected": sample.get(rejected_key, None),
        }

    return ds.map(format_sample, num_proc=kwargs.get("num_proc", 1))


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

    return ds.map(add_preference, fn_kwargs={"error_range": error_range}, num_proc=kwargs.get("num_proc", 1))


def load_audio(ds, **kwargs):
    """Post process the dataset."""

    def read_audio(sample):
        """Read audio from the file."""
        audio, sr = sf_read(sample["audio_path"])
        return {"audio": audio, "sr": sr}

    ds = ds.map(read_audio, num_proc=kwargs.get("num_proc", 1))
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
        ds = ds.filter(lambda x: x["id"] in ids, num_proc=kwargs.get("num_proc", 1))
        print(f"Filter dataset: {n_egs} to {len(ds)}")
    return ds


def add_rare_keywords(ds, **kwargs):
    tn_name = kwargs.get("tn_name", "english")
    min_len_diff = kwargs.get("min_len_diff", 2)
    common_file = kwargs.get("common_file", None)
    common_num = kwargs.get("common_num", 1000)
    assert common_file is not None, "common_file must be set"
    common_words = read_words(common_file, num=common_num, tn_name=tn_name)

    def rare_words(egs):
        text = text_norm(egs["text"], tn_name)
        words = set(text.split())
        rare_words = find_rare(words, common_words, nd=min_len_diff)
        return {
            "keywords": list(rare_words),
        }

    ds = ds.map(rare_words, num_proc=kwargs.get("num_proc", 1))
    return ds


def filter_by_keywords(ds, **kwargs):
    min_num = kwargs.get("min_num", None)
    min_ratio = kwargs.get("min_ratio", None)
    skip_none = kwargs.get("skip_none", True)
    assert (min_num is not None) or (min_ratio is not None), "Either min_num or min_ratio must be set"

    def is_enough_keywords(egs):

        keywords = egs.get("keywords", None)
        if keywords is None:
            return not skip_none
        n_keywords = len(keywords)
        if min_num is not None and n_keywords < min_num:
            return False
        elif min_ratio is not None:
            n_words = len(set(egs["text"].split()))
            ratio = len(keywords) / (n_words + 1e-6)
            if ratio < min_ratio:
                return False
        return True

    n_egs = len(ds)
    ds = ds.filter(is_enough_keywords, num_proc=kwargs.get("num_proc", 1))
    print(f"Filtered dataset: {n_egs} to {len(ds)}")
    return ds


def wer_filter_ds(ds, **kwargs):
    """Filter the dataset."""
    wer_range = kwargs.get("wer_range", None)
    bwer_range = kwargs.get("bwer_range", None)
    uwer_range = kwargs.get("uwer_range", None)

    def is_good(wer, wer_range=None):
        if wer_range is None or wer is None:
            return True
        if not isinstance(wer_range, (list, tuple)):
            wer_range = [wer_range]
        return wer_range[0] <= wer <= wer_range[-1]

    def wer_filter_fn(x):
        good = is_good(x.get("WER", None), wer_range) and is_good(x.get("BWER", None), bwer_range) and is_good(x.get("UWER", None), uwer_range)
        return good

    n_egs = len(ds)
    ds = ds.filter(wer_filter_fn, num_proc=kwargs.get("num_proc", 1))
    all_rank_print(f"Filtered dataset: {n_egs} to {len(ds)}")
    return ds


def all_rank_print(*args, **kwargs):
    rank_print(*args, main=False, **kwargs)


def dist_state():
    from accelerate import PartialState

    return PartialState()


def stream_shuffle(ds, **kwargs):
    """Process the dataset."""
    streaming = kwargs.get("streaming", False)
    if streaming:
        num_shards = kwargs.get("num_shards", dist_state().num_processes)  # this is shared with shard_ds
        ds = ds.to_iterable_dataset(num_shards=num_shards)
    num_egs = kwargs.get("num_egs", None)
    if num_egs is not None:
        ds = ds.take(num_egs)
    return ds


def shard_ds(ds, **kwargs):
    """Shard the dataset."""
    num_shards = kwargs.get("num_shards", dist_state().num_processes)
    shard_id = kwargs.get("shard_id", dist_state().process_index)
    all_rank_print(f"Sharding dataset into {num_shards} shards, picking {shard_id}")
    all_rank_print("Original dataset:", ds)
    if num_shards > 1:
        ds = ds.shard(
            num_shards=num_shards,
            index=shard_id,
            contiguous=kwargs.get("contiguous", False),  # keeps a contiguous block; set False if you prefer striding
        )
    all_rank_print("Sharded dataset:", ds)
    return ds


def path_map(ds, **kwargs):
    """Map the dataset paths."""
    field = kwargs.get("field", "audio_path")
    src_part = kwargs.get("src_part", None)
    dst_part = kwargs.get("dst_part", None)

    def map_fn(x):
        x[field] = x[field].replace(src_part, dst_part)
        return x

    if src_part and dst_part:
        ds = ds.map(map_fn, num_proc=kwargs.get("num_proc", 1))
    return ds


def post_process(ds, **kwargs):
    """Post process the dataset."""
    num_proc = kwargs.get("num_proc", 1)
    ds = stream_shuffle(ds, **kwargs)
    if path_map_kwargs := kwargs.get("path_map", {}):
        ds = path_map(ds, num_proc=num_proc, **path_map_kwargs)
    if kwargs.get("load_audio", False):
        ds = load_audio(ds, num_proc=num_proc)
    if kwargs.get("do_shard", False):
        ds = shard_ds(ds, **kwargs)
    return ds


def trunc_left_at_punc(text: str) -> str:
    """
    Truncate the string from the left at the first punctuation mark.
    Keeps the part after the punctuation, discards what’s before and including it.
    If no punctuation is found, returns the original string.
    """
    words = text.split()
    for i, word in enumerate(words):
        if word[-1] in string.punctuation:
            return " ".join(words[i + 1 :]).strip()  # cut at punctuation and strip leading spaces
    return text


def overlap_prefix(ds, **kwargs):
    """Complete the transcription for the given examples."""
    prefix_ratio = to_list(kwargs.pop("prefix_ratio", (0, 1)))
    log_interval = kwargs.get("log_interval", 10000)

    def add_overlap_prefix(egs, idx):
        words = egs["text"].split()
        ratio = random.uniform(prefix_ratio[0], prefix_ratio[-1])
        n_pfx = int(len(words) * ratio)
        prefix = " ".join(words[:n_pfx])
        prompt = f"Transcribe the audio clip into text with the prefix [{prefix}]"
        text = " ".join(words[n_pfx:])

        if idx % log_interval == 0:
            print(f"[{idx}], Prompt: {prompt}")
            print(f"[{idx}], Text  : {text}")

        return {
            "text": text,
            "prompt": prompt_format.format(prompt),
        }

    ds = ds.map(add_overlap_prefix, with_indices=True, num_proc=kwargs.get("num_proc", 1))
    return ds


def get_value(d, key, default=None):
    """Get a value from a nested dictionary using dot notation."""
    keys = key.split(".")
    for k in keys:
        if k in d:
            d = d[k]
        else:
            return default
    return d


def context_prefix(ds, **kwargs):
    """Complete the transcription for the given examples."""
    prefix_key = kwargs.get("prefix_key", "info.preceding_original_transcription")
    root_key = prefix_key.split(".")[0]  # get the root key
    prefix_range = to_list(kwargs.get("prefix_range", (0, 100)))
    log_interval = kwargs.get("log_interval", 10000)

    def add_context_prefix(egs, idx):
        pfx_words = get_value(egs, prefix_key, "").strip().split()
        n_pfx = random.randint(prefix_range[0], prefix_range[-1])
        prefix = trunc_left_at_punc(" ".join(pfx_words[-n_pfx:]))
        if prefix:
            prompt = f"Transcribe the audio clip into text with the prefix: \n{prefix}\n"
        else:
            prompt = "Transcribe the audio clip into text."

        if idx % log_interval == 0:
            print(f"[{idx}], Prompt: {prompt}")
            print(f"[{idx}], Text  : {egs['text']}")
        return {"prompt": prompt_format.format(prompt)}

    ds = ds.map(add_context_prefix, with_indices=True, remove_columns=[root_key], num_proc=kwargs.get("num_proc", 1))
    return ds


def augment(ds, **kwargs):
    """Augment the dataset with additional information."""
    num_proc = kwargs.get("num_proc", 1)
    if filter_kwargs := kwargs.get("filter", {}):
        ds = filter_ds(ds, num_proc=num_proc, **filter_kwargs)
    if wer_filter_kwargs := kwargs.get("wer_filter", {}):
        ds = wer_filter_ds(ds, num_proc=num_proc, **wer_filter_kwargs)
    if overlap_prefix_kwargs := kwargs.get("overlap_prefix", {}):
        ds = overlap_prefix(ds, num_proc=num_proc, **overlap_prefix_kwargs)
    if context_prefix_kwargs := kwargs.get("context_prefix", {}):
        ds = context_prefix(ds, num_proc=num_proc, **context_prefix_kwargs)
    if biasing_kwargs := kwargs.get("biasing", {}):
        ds = bias_sampling(ds, num_proc=num_proc, **biasing_kwargs)
    if pref_kwargs := kwargs.get("simu_preference", {}):
        ds = simulate_preference(ds, num_proc=num_proc, **pref_kwargs)
    if fmt_pref_kwargs := kwargs.get("format_preference", {}):
        ds = format_preference(ds, num_proc=num_proc, **fmt_pref_kwargs)
    if add_rare_keywords_kwargs := kwargs.get("add_rare_keywords", {}):
        ds = add_rare_keywords(ds, num_proc=num_proc, **add_rare_keywords_kwargs)
    if filter_by_keywords_kwargs := kwargs.get("filter_by_keywords", {}):
        ds = filter_by_keywords(ds, num_proc=num_proc, **filter_by_keywords_kwargs)
    if post_process_kwargs := kwargs.get("post_process", {}):
        ds = post_process(ds, num_proc=num_proc, **post_process_kwargs)
    return ds


def cache_ds(**kwargs):
    cache_name = kwargs.get("cache_name", None)
    if cache_name is None:
        return None, None
    if cache_name == "auto":
        config_path = get_config_path()  # ensure config path is set
        assert config_path is not None, "config_path must be set for auto cache_name"
        cache_name = Path(config_path).stem
    cache_dir = kwargs.get("cache_dir", Path().home() / "data/cache_datasets")
    cache_path = Path(cache_dir) / cache_name
    ds = load_cached_ds(cache_path)
    return ds, cache_path


def load_cached_ds(cache_path):
    if not cache_path:
        return None
    try:
        rank_print(f"Loading cached dataset from {cache_path}")
        ds = Dataset.load_from_disk(cache_path)
        return ds
    except Exception as e:
        rank_print(f"Cache not found or invalid at {cache_path}, will create a new one. Error: {e}")
        return None


def create_audio_dataset(**kwargs):
    """Create a dataset from the given split."""
    ds_name = kwargs.get("dataset_name", "unknown").lower()
    with dist_state().local_main_process_first():
        ds, cache_path = cache_ds(**kwargs)
        if ds is not None:
            return ds
        if ds_name == "ls_bias":
            ds = ls_bias_dataset(**kwargs)
        elif ds_name == "inhouse_entity":
            ds = entity_dataset(**kwargs)
        elif ds_name == "openasr":
            ds = openasr_dataset(**kwargs)
        elif ds_name == "tsv":
            ds = tsv_dataset(**kwargs)
        elif ds_name == "jsonl":
            ds = jsonl_dataset(**kwargs)
        elif ds_name == "chunk":
            ds = chunk_dataset(**kwargs)
        elif ds_name == "cached":
            ds = load_cached_ds(kwargs.get("cache_path", None))
            assert ds is not None, "Cached dataset not found."
        else:
            raise ValueError(f"Unknown dataset name: {ds_name}")
        ds = augment(ds, **kwargs)
        if cache_path:
            rank_print(f"Saving dataset to cache at {cache_path}")
            ds.save_to_disk(cache_path)
    return ds


# %%
if __name__ == "__main__":
    # Example usage
    # dataset = create_dataset(name="openasr", head=2)
    # print(dataset)
    # print(dataset[0]["text"])
    tsv_path = "/datablob1/users/ruchaofan/wavllm_data/wavllm/converted_path_train_data_4chunk/asr_train_transcribe.tsv"
    import yaml

    yaml_file = Path("/mnt2/newhome/boren/trl/orng_conf/biasing/debug/dpo_bias_debug_local.yaml")
    kwargs = yaml.safe_load(yaml_file.read_text())
    ds = create_audio_dataset(**kwargs["train_data"])
    print(ds)
    print(next(iter(ds)))

# %%
