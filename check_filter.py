# %%

from pathlib import Path
from datasets import load_dataset
from pprint import pprint
import pandas as pd


def collect_info(egs):
    gt_words = egs.get("ground_truth", [])
    pt_words = egs.get("distractors", [])
    hit_words = set(gt_words) & set(pt_words)

    return {
        "n_gt": len(gt_words),
        "n_pt": len(pt_words),
        "n_hit": len(hit_words),
    }


def measure_df(ds):
    n_gt, n_pt, n_hit = 0, 0, 0
    n_utt = 0
    n_utt_hit = 0
    for egs in ds:
        n_utt += 1
        n_gt += egs["n_gt"]
        n_pt += egs["n_pt"]
        n_hit += egs["n_hit"]
        if egs["n_hit"] == egs["n_gt"]:
            n_utt_hit += 1

    return {
        "n_gt": n_gt,
        "n_pt": n_pt,
        "n_hit": n_hit,
        "recall": n_hit / n_gt if n_gt > 0 else 0.0,
        "precision": n_hit / n_pt if n_pt > 0 else 0.0,
        "n_utt": n_utt,
        "n_utt_hit": n_utt_hit,
        "utt_recall": n_utt_hit / n_utt if n_utt > 0 else 0.0,
    }


def measure_jsonl(jsonl_path):
    ds = load_dataset("json", data_files=str(jsonl_path))["train"]
    ds = ds.map(collect_info, remove_columns=ds.column_names)
    return measure_df(ds)


jsonl_paths = [
    "/home/boren/data/librispeech_biasing/ref/test-other.biasing_100.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_100.jsonl.th0.5.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_100.jsonl.th0.68.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_100.jsonl.th0.95.jsonl",
    "/home/boren/data/librispeech_biasing/ref/test-other.biasing_1000.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_1000.jsonl.th0.5.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_1000.jsonl.th0.68.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_1000.jsonl.th0.95.jsonl",
    "/home/boren/data/librispeech_biasing/ref/test-other.biasing_2000.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_2000.jsonl.th0.5.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_2000.jsonl.th0.68.jsonl",
    "/home/boren/data/librispeech_biasing/sunit_filter/20250919/test-other.biasing_2000.jsonl.th0.95.jsonl",
]


def collect_metrics(jsonl_paths, output_tsv=None):
    metrics = {}
    for jsonl_path in jsonl_paths:
        jsonl_path = Path(jsonl_path)
        metrics[jsonl_path.stem] = measure_jsonl(jsonl_path)
    df = pd.DataFrame.from_dict(metrics, orient="index").T
    if output_tsv:
        df.to_csv(output_tsv, sep="\t")
    else:
        pprint(df)
    return df


# %%
output_tsv = "/home/boren/data/librispeech_biasing/sunit_filter/20250919/filter_metrics.tsv"
df = collect_metrics(jsonl_paths, output_tsv=output_tsv)
# %%
