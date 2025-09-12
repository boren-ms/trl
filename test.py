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
