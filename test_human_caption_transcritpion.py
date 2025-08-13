# %%
import json
from pathlib import Path
from trl.scripts.chunk_dataset import load_data_from_chunk

chunk_dir = Path("/datablob/am_data/en/data_prep_e2e_R15/human_caption_v2/batch_gpt_post_process/FY22_AdjustBoundary_BiasLM_verbatim/ChunkFiles/")
chunk_name = "chunk_8076b1ca-d798-11ed-8321-002248bd2805"

# %%

meta = json.loads((chunk_dir / (chunk_name + ".json")).read_text())
count = meta["fileInfo"][0]["count"]
# %%
chunk_path = chunk_dir / (chunk_name + ".info")
info_list = load_data_from_chunk(chunk_path, "info", count)

# %%

info_egs = info_list[0]
# %%
from pprint import pprint

pprint(info_egs)
# %%
for key, value in info_egs["alternative_transcription"].items():
    print(f"{key}: {value}")
