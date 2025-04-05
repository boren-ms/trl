# %%
from datasets import load_dataset

dataset_path = "hf-audio/esb-datasets-test-only-sorted"
dataset = "librispeech"
split = "test.clean"
data = load_dataset(dataset_path, dataset, split=split)
print(len(data))

# %%
