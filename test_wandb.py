# %%
import pandas as pd
import wandb
import fire
import os
import urllib.parse
from pathlib import Path


host = os.environ.get("WANDB_ORGANIZATION", "https://msaip.wandb.io")
entity = os.environ.get("WANDB_ENTITY", "genai")
project = os.environ.get("WANDB_PROJECT", "biasing")
key = os.environ.get("WANDB_API_KEY", "")
print(f"Using W&B : {host}/{entity}/{project}")
wandb.login(host=host, key=key, relogin=True)


def get_run(path):
    parsed = urllib.parse.urlparse(path)
    path = parsed.path.replace("/runs/", "/").strip("/")
    assert path is not None, "Either url or path must be provided to get_run"
    try:
        return wandb.Api().run(path)
    except wandb.Error:
        return None


# %%
run = get_run("https://msaip.wandb.io/genai/biasing/runs/tdyghemb?nw=nwuserboren")
# %%

import pprint

pprint.pprint(run.json_config)
# %%
import json

conf_json = "config.json"
json.dump(run.config, open(conf_json, "w"), indent=2)

# %%
