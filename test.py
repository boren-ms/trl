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
