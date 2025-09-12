#! /usr/bin/env python
# -*- coding: utf-8 -*-
# %%
from transformers import pipeline
import re


def complete_prefix(text, prefix, case_sensitive=False):
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = r"\b" + re.escape(prefix) + r"\w*\b"
    matches = re.findall(pattern, text, flags)
    return matches[0] if matches else prefix


model_id = "Jean-Baptiste/roberta-large-ner-english"
ner = pipeline("ner", model=model_id, aggregation_strategy="simple")
text = "It's a high level airbender move, with some spiritual stuff thrown in. That actually made me laugh a little because during the Korra Book 2 dvd commentaries. Mike and Bryan talked about Jinora's Raava rescue. They explained that they didn't know what she was doing. They just wanted to see her resuscitate the little piece of Raava that was inside UnaVaatu. But for all intents and purposes, think of the Air substyle as Spirit Bending. Not to"
results = ner(text)


# %%
def find_entity(text, ner):
    words = [""]
    last_e = 0
    for res in ner(text):
        s, e = res["start"], res["end"]
        if not text[last_e:s].strip():
            words[-1] += text[last_e:e]
        else:
            words.append(text[s:e])
        last_e = e
    words = set([w.strip() for w in words if w.strip()])
    return words


entities = find_entity(text, ner)
print(entities)

# %%
# entities = set()
# for segment in ner(text):
#     word = segment["word"].strip()
#     word = complete_prefix(text, word)
#     entities.update(word.split())

# %%
