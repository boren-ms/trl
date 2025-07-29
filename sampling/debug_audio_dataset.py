# %%

from pathlib import Path
import soundfile as sf
import sys
import random
from collections import defaultdict
from typing import List, Dict, Any, Iterator

paths = [str(x) for x in Path(__file__).parents[:2]]
sys.path.extend(paths)
from trl.scripts.chunk.chunker import Chunker

# %%
audio_path = Path("/home/boren/data/Evaluation/InhouseASR/EWER/en-US-entity-v3/CustomerSpeechDomainSet_DTEST_Medical_Entity_FY23Q4_en-US_DTEST/wav/bca0f4fb-f005-47d4-9b9c-c9de52edf521_0.wav")
# %%
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import torch
import string


def generate_random_string(length: int) -> str:
    """Generate a random string of specified length."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


class DummyDataset:
    """Simple dataset with random character strings."""

    def __init__(self, num_samples: int, min_length: int = 5, max_length: int = 50):
        self.data = [generate_random_string(random.randint(min_length, max_length)) for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return self.data[idx]


class LengthBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups data by length."""

    def __init__(self, dataset: DummyDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        sizes = torch.tensor([len(x) for x in self.dataset])
        for batch in torch.chunk(torch.argsort(sizes), len(self)):
            yield batch.tolist()


# %%
# Create dummy dataset and data loader
dataset = DummyDataset(num_samples=100, min_length=5, max_length=50)
batch_sampler = LengthBatchSampler(dataset, batch_size=8)
dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

# Demonstrate usage
print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")

for i, batch in enumerate(dataloader):
    if i < 3:  # Show first 3 batches
        lengths = [len(item) for item in batch]
        print(f"Batch {i}: sizes {lengths}, items: {batch}")
    else:
        break

# %%
