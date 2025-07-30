# %%

from pathlib import Path
import soundfile as sf

# %%
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import torch
import string
from datasets import load_dataset, concatenate_datasets

jsonl_path = Path("/home/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3/CustomerSpeechDomainSet_DTEST_Banking_Entity_FY23Q4_en-US_DTEST/test.jsonl")

ds = load_dataset("json", data_files=[str(jsonl_path)], split="train")

ds[0]
from bs4 import BeautifulSoup

# %%
text = ds[0]["Transcription"]
bs = BeautifulSoup(text, "html.parser")
entities = [tag.get_text().strip() for tag in bs.find_all() if tag.name.startswith("ne")]

# %%
import re

# Extract clean text content from the parsed HTML
# clean_text = bs.get_text()
clean_text = text
# Split text into utterances by punctuation (.!?;)
utterances = re.split(r"[.!?;]+", clean_text)

# Clean up utterances: strip whitespace and filter out empty ones
utterances = [utterance.strip() for utterance in utterances if utterance.strip()]

print(f"Number of utterances: {len(utterances)}")
for i, utterance in enumerate(utterances):
    print(f"{i+1}: {utterance}")

# %%
