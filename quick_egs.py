#%%
import datasets
#%%

data_path="/home/boren/data/gsm8k"
dataset = datasets.load_dataset(data_path, data_files={"train": "train.parquet", "test": "test.parquet"}, split="train")
print(dataset)
# %%
