# %%
from trl.scripts.audio_dataset import create_audio_dataset


# %%
common_file = "/home/boren/data/librispeech_biasing/words/all_words.count.txt"
conf = {
    "dataset_name": "chunk",
    # "max_chunks": 2,
    # "max_egs": 30,
    "num_proc": 20,
    "specs": [
        "/home/boren/data/inhouse/data_spec/asr_chunk_inhouse_en_fy22.json",
    ],
    "add_rare_keywords": {
        "common_file": common_file,
        "common_num": 3000,
    },
    "filter_by_keywords": {
        "min_ratio": 0.1,
    },
    "cache": True,
    "cache_tag": "asr_chunk_inhouse_en_fy22_sc3k_0.1r",
}

ds = create_audio_dataset(**conf)
for i, egs in enumerate(ds):
    n_words = len(egs["text"].split())
    n_keywords = len(egs["keywords"])
    print("index:", i)
    print("text:", egs["text"])
    print("keywords:", egs["keywords"])
    print(f"num_words: {n_words}, num_keywords: {n_keywords}, ratio: {n_keywords/(n_words+1e-6):.3f}")
    print()
    if i >= 10:
        break

# %%
