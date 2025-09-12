# pip install ray[default] transformers datasets torch
from attrs import field
import ray
import torch
from itertools import chain
from more_itertools import divide
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset


def join_entities(outputs, text):
    entities = [""]
    last_e = 0
    for output in outputs:
        s, e = output["start"], output["end"]
        if not text[last_e:s].strip():
            entities[-1] += text[last_e:e]
        else:
            entities.append(text[s:e])
        last_e = e
    entities = list(set([w.strip() for w in entities if len(w.strip()) > 1]))  # remove empty and single char
    return entities


@ray.remote(num_gpus=1)
class NERActor:
    def __init__(self, model_id, device=0):
        self.pipe = pipeline(
            "ner",
            model=model_id,
            aggregation_strategy="simple",
            device=device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def infer_batch(self, ds, bs=8, key="text"):
        outputs = []
        ds = KeyDataset(ds, key)
        for i, output in enumerate(self.pipe(ds, batch_size=bs)):
            entities = [x["word"] for x in output]
            outputs.append(entities)
        return outputs


_ACTORS = None


def _get_actors(model_id, num_actors=0):
    global _ACTORS
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    if _ACTORS is None:
        n_gpus = int(ray.available_resources().get("GPU", 1))
        print(f"Ray detected {n_gpus} GPUs")
        if num_actors <= 0:
            num_actors = n_gpus
        gpu_per_actor = max(min(1.0, n_gpus / num_actors), 0.1)
        task = NERActor.options(num_gpus=gpu_per_actor)
        print(f"Creating {num_actors} NER actors on {n_gpus} GPUs, {gpu_per_actor} GPU per actor")
        _ACTORS = [task.remote(model_id, i % n_gpus) for i in range(num_actors)]
    return _ACTORS


def ray_ner(texts, model_id, num_actors=0):
    actors = _get_actors(model_id, num_actors)
    futures = []
    for i, chunk in enumerate(divide(len(actors), texts)):
        chunk = list(chunk)
        futures.append(actors[i].infer_batch.remote(chunk))
    results = ray.get(futures)
    keywords = list(chain(*results))
    return keywords


def ner_ds(ds, model_id, src_field="text", tgt_field="keywords", bs=None, n_actors=0):
    actors = _get_actors(model_id, n_actors)
    futures = []
    n = len(actors)
    for i in range(n):
        ds_i = ds.shard(num_shards=n, index=i, contiguous=True)
        futures.append(actors[i].infer_batch.remote(ds_i, bs=bs, key=src_field))
    results = ray.get(futures)
    ds = ds.add_column(tgt_field, list(chain(*results)))
    return ds


def ner_map(batch, **kwargs):
    texts = batch["prompt"]
    if not texts:
        return {"keywords": []}
    num_actors = kwargs.get("num_actors", 0)
    model_path = kwargs.get("model_path", None)
    assert model_path is not None, "Please provide a valid model_path"
    keywords = ray_ner(texts, model_path, num_actors)
    return {"keywords": keywords}


if __name__ == "__main__":
    # Example: load your HF dataset
    model_path = "/home/boren/data/ckp/hf_models/roberta-large-ner-english/"
    n_actors = 2
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    ds2 = ner_ds(ds, model_path, n_actors, key="prompt")
    # ds2 = ds.map(
    #     ner_map,
    #     fn_kwargs={"model_path": model_path, "num_actors": n_actors},
    #     batched=True,
    #     num_proc=1,
    #     desc="NER via Ray RPC",
    # )
    print(ds2)
    print(ds2[0])

    # Example usage:
    # result = ner_map(batch, n=0)
