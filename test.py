#! /usr/bin/env python
# -*- coding: utf-8 -*-
# %%
from transformers import pipeline

model_id = "Jean-Baptiste/roberta-large-ner-english"
ner = pipeline("ner", model=model_id, aggregation_strategy="simple")
texts = [
    "It's a high level airbender move, with some spiritual stuff thrown in.",
    " That actually made me laugh a little because during the Korra Book 2 dvd commentaries. ",
    "Mike and Bryan talked about Jinora's Raava rescue. They explained that they didn't know what she was doing. They just wanted to see her resuscitate the little piece of Raava that was inside UnaVaatu. But for all intents and purposes, think of the Air substyle as Spirit Bending. Not to",
]

results = ner(texts)


# %%
# entities = set()
# for segment in ner(text):
#     word = segment["word"].strip()
#     word = complete_prefix(text, word)
#     entities.update(word.split())

# %%
import pandas as pd

from pathlib import Path

# %%
data_dir = Path("~/data/librispeech_biasing/sunit_filter/20250919/").expanduser()
raw_jsonl = data_dir / "metric_vllm_ls_other_2000__results.jsonl"
sunit_jsonl = data_dir / "metric_vllm_ls_other_2000_sunit_th05__results.jsonl"

df_raw = pd.read_json(raw_jsonl, lines=True)
df_sunit = pd.read_json(sunit_jsonl, lines=True)
# %%

df = df_raw.merge(
    df_sunit,
    on=["id"],
    suffixes=("_raw", "_sunit"),
)


# %%
mdf = df[["id", "hyp_raw", "hyp_sunit", "prompt_raw", "prompt_sunit"]]
from datasets import Dataset


ds = Dataset.from_pandas(mdf)


# %%
def extract_words(text):
    text = text.replace("<|user|><|audio_1|>Transcribe the audio clip into text. Pay extra attention to the following phrases/words. ", "")
    text = text.replace("<|end|><|assistant|>", "")
    return [w.strip().strip("*") for w in text.split(",")]


def process(example):
    input_words_raw = extract_words(example["prompt_raw"])
    input_words_sunit = extract_words(example["prompt_sunit"])
    diff = set(input_words_sunit) - set(input_words_raw)
    return {
        "diff": diff,
        "input_words_raw": input_words_raw,
        "input_words_sunit": input_words_sunit,
    }


ds = ds.map(process)
# %%
n_total = len(ds)
n_sunit = 0
n_raw = 0
for example in ds:
    n_sunit += len(example["input_words_sunit"])
    n_raw += len(example["input_words_raw"])
print(f"Average input words: raw {n_raw/n_total:.2f}, sunit {n_sunit/n_total:.2f}")
# %%
