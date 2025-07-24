# %%
import pandas as pd
from pathlib import Path


json_path = Path(__file__).parent / "test.json"
df = pd.read_json(json_path)
filtered_df = df[df["WER"] > 0]

# Example usage:
# filtered = load_and_filter_json(test_json)
# print(filtered)

# %%
