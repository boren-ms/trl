# %%
from pathlib import Path
from collections import Counter
import json


def load_words(word_path):
    wd_cnt = {}
    with open(word_path, "r", encoding="utf-8") as f:
        for line in f:
            word, count = line.rsplit("\t", 1)
            wd_cnt[word] = int(count)
    return wd_cnt


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


import re


def to_set(words):
    wd_set = set()
    for w in words:
        wds = re.split(r"\W+", w)
        wd_set.update(wds)
    return wd_set


# %%
# word_file = Path("/home/boren/data/cache_datasets/stage1_en_asr_hc_tag_entity/grpo_rare_hc_tag_entity_cache_zero_e1_bp8_ref_fn1.words.txt")
word_file = Path("/home/boren/data/cache_datasets/stage1_en_asr_hc_tag_entity/grpo_rare_hc_tag_entity_cache_zero_e1_bp8_ref_fn1.keywords.txt")
wd_cnt = load_words(word_file)

kwd_file = Path("/home/boren/data/cache_datasets/entity_statics//grpo_rare_debug_local.entity_Gaming_0.keywords.json")
kwd_cnt = load_json(kwd_file)


# %%
def check_coverage(keywords, words):
    kwd_set = to_set(keywords)
    wd_set = to_set(words)
    diff = kwd_set - wd_set
    print("Number of words:", len(wd_set))
    print("Number of keywords:", len(kwd_set))
    print("Number of diff:", len(diff))
    print("Diff:", diff)
    print("Coverage: {:.2f}%".format((1 - len(diff) / len(kwd_set)) * 100))


# %%
word_file = Path("/home/boren/data/cache_datasets/stage1_en_asr_hc_tag_entity/grpo_rare_hc_tag_entity_cache_zero_e1_bp8_ref_fn1.keywords.txt")
word_file = Path("/home/boren/data/cache_datasets/stage1_en_asr_hc_tag_entity/grpo_rare_hc_tag_entity_cache_zero_e1_bp8_ref_fn1.words.txt")
wd_cnt = load_words(word_file)
# %%
kwd_file = Path("/home/boren/data/cache_datasets/entity_statics//grpo_rare_debug_local.entity_Insurance_0.keywords.json")
kwd_file = Path("/home/boren/data/cache_datasets/entity_statics//grpo_rare_debug_local.entity_Gaming_0.keywords.json")
kwd_cnt = load_json(kwd_file)

print("File:", kwd_file.stem)
check_coverage(kwd_cnt.keys(), wd_cnt.keys())


# %%
# _rare_debug_local.entity_Gaming_0.keywords
# Number of words: 3414121
# Number of keywords: 1841
# Number of diff: 43
# Diff: {'DKLE', 'Wovama', 'overwatched', 'iMICE', 'PeohZarr', 'Ithecles', 'megazords', 'MegaRetroN', 'blorbos', 'BatteryJack', 'Ochaking', 'upsampling', 'Nitrosyl', 'jumpable', 'installers', 'BloodBot', 'Korosu', 'Runmus', 'Pixelplus', 'siccoopertwitch', 'Afterburner2', 'simulacrum', 'playerbase', 'Battery1', 'microtransactions', 'SeTa', 'platformers', 'BloodBat', 'footsies', 'Fazerblaster', 'homebrews', 'Hinokuri', 'DarkSector', 'platforming', 'Picktech', 'siccooper', 'icebergs', 'securitybreachtv', 'Toastybros', 'Bobbiedots', 'Belokk', 'Husaria', 'WOVAMA'}
# Coverage: 97.66%
# File: grpo_rare_debug_local.entity_Insurance_0.keywords
# Number of words: 3414121
# Number of keywords: 621
# Number of diff: 28
# Diff: {'floodsmart', 'HASs', 'coinsurance', 'dismemberment', 'DE2525XX', 'FloodTools', 'cancelable', 'underinsured', 'AssetVault', 'copays', 'CPARA', 'floodtools', 'BetterWealth', 'coverages', '0344', 'deductibles', 'DE2501', 'Deductibles', 'daviddufourd', 'insurable', 'moneyguy', 'MGAI', 'DE2547', 'iHealthBrokers', 'DE2547A', 'Insurable', 'insurability', '4178821800'}
# Coverage: 95.49%
