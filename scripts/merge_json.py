# %%
from pathlib import Path
import json
from collections import defaultdict


def merge_json_files(chunk_dir: Path, output_file: Path):
    json_files = list(chunk_dir.glob("*.json"))
    all_data = defaultdict(list)
    for json_file in json_files:
        data = json.load(open(json_file, "r"))
        all_data["fileType"] = data["fileType"]  # only one fileType
        all_data["fileInfo"].extend(data.get("fileInfo", []))

    with open(output_file, "w") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    return output_file


def merge_jsons_in_dir(data_dir: Path):
    chunk_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    for chunk_dir in chunk_dirs:
        output_file = data_dir / f"{chunk_dir.name}_merged.json"
        merge_json_files(chunk_dir, output_file)
        print(f"Merged JSON saved to: {output_file}")


# %%

data_dir = Path("/home/boren/data/inhouse/entity_data/entity_chunk")

chunks = ["Insurance_16k", "Gaming_43k", "K12_Higher_Education_36k", "Retail_26k", "Science_And_Tech_48k"]
for chunk in chunks:
    chunk_dir = data_dir / chunk / "ChunkFiles"
    merge_jsons_in_dir(chunk_dir)
# %%
