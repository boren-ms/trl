#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from collections import Counter
from datasets import Dataset
import yaml
import fire
from trl.scripts.audio_dataset import create_datasets


def yaml_print(data, **kwargs):
    print(yaml.dump(data, sort_keys=False, default_flow_style=False), **kwargs)


def load_datasets(conf_path, field="train_data", cache_name=None):
    with open(conf_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    ds_conf = conf.get(field, {})
    if not ds_conf:
        raise ValueError(f"No dataset config found for '{field}' in {conf_path}")
    if cache_name:
        ds_conf["cache_name"] = cache_name
    print("Loading DS:")
    yaml_print(ds_conf)
    datasets = create_datasets(**ds_conf)
    if not isinstance(datasets, dict):
        datasets = {field: datasets}
    print("Dataset loaded. Summary:")
    for key, value in datasets.items():
        print(f"{key}:\n{value}")

    return datasets


def collect_info(ds):
    wd_cnt = Counter()
    kwd_cnt = Counter()
    for egs in ds:
        text = egs.get("text", "")
        words = [w for w in text.split() if w]
        wd_cnt.update(words)
        keywords = egs.get("keywords", [])
        keywords = [kw for kw in keywords if kw]
        kwd_cnt.update(keywords)
    return wd_cnt, kwd_cnt


def write_counts(wd_cnt, wd_path):
    with open(wd_path, "w", encoding="utf-8") as f:
        for word, count in wd_cnt.most_common():
            f.write(f"{word}\t{count}\n")
    print("Counts written to:", wd_path)


def dump_ds_info(ds, output_path):
    wd_cnt, kwd_cnt = collect_info(ds)
    word_path = output_path.with_suffix(".words.txt")
    keyword_path = output_path.with_suffix(".keywords.txt")
    write_counts(wd_cnt, word_path)
    write_counts(kwd_cnt, keyword_path)
    info_dict = {
        "config": str(output_path),
        "total_samples": len(ds) if isinstance(ds, Dataset) else "unknown",
        "ds_columns": ds.column_names,
        "total_words": sum(wd_cnt.values()),
        "unique_words": len(wd_cnt),
        "word_count_file": str(word_path),
        "total_keywords": sum(kwd_cnt.values()),
        "unique_keywords": len(kwd_cnt),
        "keyword_count_file": str(keyword_path),
    }
    print("Dataset info:")
    yaml_print(info_dict)
    info_path = output_path.with_suffix(".info.yaml")
    print("Info written to:", info_path)
    with open(info_path, "w", encoding="utf-8") as f:
        yaml_print(info_dict, file=f)


def main(config, field=None, cache_name=None):
    config = Path(config)
    ds_dict = load_datasets(config, field=field, cache_name=cache_name)
    for name, ds in ds_dict.items():
        print(f"Processing dataset: {name}")
        output_path = config.with_suffix(f".{name}")
        dump_ds_info(ds, output_path)


if __name__ == "__main__":
    fire.Fire(main)
