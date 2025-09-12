#! /usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing as mp

mp.set_start_method("spawn", force=True)
from trl.scripts.audio_dataset import create_audio_dataset
from ray_tool import sync_remote_dir
import yaml
from pathlib import Path
import fire


def proc_dataset(conf_path, push_remote=False):
    conf_path = Path(conf_path)
    conf = yaml.safe_load(conf_path.read_text())["train_data"]
    conf.update({"cache_name": conf_path.stem})

    print("Config:")
    print(yaml.dump(conf, sort_keys=False, default_flow_style=False))

    ds = create_audio_dataset(**conf)
    print("Got dataset info")
    print(ds)
    print()
    cache_path = f"~/data/cache_datasets/{conf_path.stem}"
    if push_remote:
        print(f"Syncing dataset [{cache_path}] to remote storage...")
        sync_remote_dir(cache_path, True)
    print("dataset sample [0]:")
    egs = ds[0]
    n_words = len(egs["text"].split())
    n_keywords = len(egs["keywords"])
    print("text:", egs["text"])
    print("keywords:", egs["keywords"])
    print(f"num_words: {n_words}, num_keywords: {n_keywords}, ratio: {n_keywords/(n_words+1e-6):.3f}")


if __name__ == "__main__":

    fire.Fire(proc_dataset)
    # conf_path = "orng_conf/biasing/data/ls_sc1k_fr01.yaml"
    # proc_dataset(conf_path)
