#! /usr/bin/env python
# -*- coding: utf-8 -*-
# %%
from trl.scripts.shared_utils import create_dataset
from trl.scripts.audio_metrics import text_norm
import yaml
from pathlib import Path
from collections import Counter

from trl.scripts.audio_metrics import text_norm

# %%
# conf_path = "orng_conf/biasing/data/ls_sc1k_fr01.yaml"
conf_path = "orng_conf/biasing/data/entity_set.yaml"
conf_path = Path(conf_path)
conf = yaml.safe_load(conf_path.read_text())["train_data"]
print("Config:")
print(yaml.dump(conf, sort_keys=False, default_flow_style=False))
ds = create_dataset(conf)
print("Got dataset info")
print(ds)
if isinstance(ds, dict):
    ds = next(iter(ds.values()))
num_egs = len(ds)
word_counts = Counter()
total_words = 0
word_counts_per_example = []
keywords = set()
for egs in ds:
    # You may need to adjust 'transcript' to match your dataset's field name
    text = text_norm(egs.get("text", ""))
    words = text.split()
    word_counts.update(words)
    total_words += len(words)
    word_counts_per_example.append(len(words))
    keywords.update(egs.get("keywords", []))


num_unique_words = len(word_counts)
avg_words_per_example = total_words / num_egs if num_egs else 0
sorted_counts = sorted(word_counts_per_example)
n = len(sorted_counts)

median_words = sorted_counts[n // 2]
word_cnt_path = conf_path.with_suffix(".wd_cnt.txt")
with open(word_cnt_path, "w", encoding="utf-8") as f:
    for word, count in word_counts.most_common():
        f.write(f"{word} {count}\n")
keyword_cnt_path = conf_path.with_suffix(".kw_cnt.txt")
with open(keyword_cnt_path, "w", encoding="utf-8") as f:
    for keyword in sorted(keywords):
        f.write(f"{keyword}\n")

info_dict = {
    "num_egs": num_egs,
    "num_unique_words": num_unique_words,
    "num_unique_keywords": len(keywords),
    "avg_words_per_example": avg_words_per_example,
    "median_words": median_words,
    "word_count_file": str(word_cnt_path),
    "keyword_count_file": str(keyword_cnt_path),
}

info_file_path = conf_path.with_suffix(".info.yaml")
with open(info_file_path, "w", encoding="utf-8") as f:
    yaml.dump(info_dict, f, sort_keys=False, default_flow_style=False)

print("Dataset info written to:", info_file_path)
print(yaml.dump(info_dict, sort_keys=False, default_flow_style=False))

# %%
