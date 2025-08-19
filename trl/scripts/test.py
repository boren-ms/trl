# %%
from trl.scripts.chunk_dataset import *
from pathlib import Path

# %%
chunk_path = Path("/datablob1/users/xioxiao/am_data/en_long/ami_ihm_mix_profiles_v2/train_dev/chunks/chunk_b7a4d20c-7283-11f0-9682-b04f13015df5.json")

import re

info_data = load_chunk_info(str(chunk_path))
# %%
chunk = {
    "chunk_path": str(chunk_path.parent),
    **info_data[0],
}

examples = load_examples(chunk, ["audio", "info"])

# %%
egs = examples[0]
# %%
spk_trans = egs["info"]["alternative_transcription"]["display_human_caption_adjusted_with_spk_id_tag"]


def merge_speakers(text):
    cur_spk = None
    words = []
    for wd in text.split():
        if re.match(r"<SPK\d+>", wd):
            if cur_spk == wd:
                continue
            else:
                cur_spk = wd
        words.append(wd)
    return " ".join(words)


# %%
specs = [
    {
        "manifest_file": str(chunk_path),
        "chunk_path": str(chunk_path.parent),
    }
]
for egs in generate_examples(specs, ["audio", "info"]):
    print(egs)
    break
# %%
egs["info"]["alternative_transcription"]["display_human_caption_adjusted_with_spk_id_tag"]

# %%
