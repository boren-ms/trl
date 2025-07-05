#%%
from trl.scripts.audio_dataset import ls_bias_dataset
from pathlib import Path

home_dir = Path.home()
REMOTE_DIR="az://orngscuscresco/data/boren/data"

data_dir = home_dir / "data/librispeech_biasing/ref"
data_paths = [
    data_dir / "test-clean.biasing_100.jsonl",
    # data_dir / "test-clean.biasing_500.jsonl",
    # data_dir / "test-clean.biasing_1000.jsonl",
]
ds = ls_bias_dataset(data_paths, bias_key="distractors", tag=True, num_egs=10, data_dir=REMOTE_DIR)

for i, sample in enumerate (ds):
    print(f"Sample {i}:")  # noqa
    print("Prompt", sample["prompt"][0]["content"])  # noqa
    print("Text:", sample["text"])  # noqa
    print("Audio Path:", sample["audio_path"])
    break