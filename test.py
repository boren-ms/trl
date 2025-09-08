# %%

import random


def rand_int(min_val, max_val):
    """Randomly sample an integer between min_val and max_val."""
    mu = (min_val + max_val) / 2
    sigma = (max_val - min_val) / 6
    return int(random.gauss(mu, sigma))


print("Random number:", rand_int(0, 1000))
# %%
y = [rand_int(0, 1000) for i in range(10)]
# %%
print("List of random numbers:", y)
# %%
print("mean:", sum(y) / len(y))
print("max:", max(y))
print("min:", min(y))
# %%
ds_path = "/home/boren/data/cache_datasets/asr_chunk_inhouse_en_sc3k_0.1r"
from datasets import load_from_disk

ds = load_from_disk(ds_path)

print(ds)
# %%
for i, egs in enumerate(ds):

    keywords = egs.get("keywords", None)
    words = set(egs.get("text", None).split())
    print(" audio_chunk:", egs["audio_chunk"])
    print("  keywords:", keywords)
    print("  keywords rate:", len(keywords) / len(words) if words else 0)
    print("  words:", words)
    if i > 10:
        break

# %%
