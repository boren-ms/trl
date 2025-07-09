# %%
from datasets import load_dataset
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from trl.scripts.audio_dataset import create_audio_dataset

# 1. Load a dataset (e.g., IMDB)
# dataset = load_dataset("imdb", split="train")
# %%
data_dir = Path("/home/boren/data/librispeech_biasing/ref")
jsonl_path = data_dir / "test-clean.biasing_100.jsonl"
dataset = create_audio_dataset(dataset_name="ls_bias", jsonl_path=str(jsonl_path))
# %%
dataset = dataset.select_columns(["prompt", "audio_path", "trans"])
# %%
# 3. Use DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. Iterate through batches
for batch in dataloader:
    print(batch)
    break

# %%
