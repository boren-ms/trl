# %%
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
from more_itertools import chunked
from trl.extras.vllm_client import VLLMClient
from trl.scripts.audio_dataset import ls_bias_dataset
from trl.scripts.audio_metrics import compute_wers
from trl.data_utils import sf_read

# ----------------- Config -----------------
home_dir = Path.home()

REMOTE_DIR = "az://orngscuscresco/data/boren/data"

host = "10.139.241.19"
n_gen = 1
max_tokens = 512
temperature = 0.0
top_p = 1.0
top_k = -1
min_p = 0.0
repetition_penalty = 1.0
device = "cuda:7"


def prepare_dataset(data_path, bias_key="distractors", num_egs=None):
    ds = ls_bias_dataset(str(data_path), bias_key=bias_key, tag=True, data_dir=REMOTE_DIR)
    if num_egs is not None:
        ds = ds.select(range(num_egs))
    fmt = "<|user|>{}<|end|><|assistant|>"
    inputs = {}
    for sample in ds:
        id = Path(sample["audio_path"]).stem
        inputs[id] = {"prompt": fmt.format(sample["prompt"][0]["content"]), "audio": sample["audio_path"], "ref": sample["text"]}
    prompts = [val["prompt"] for val in inputs.values()]
    audios = [val["audio"] for val in inputs.values()]
    ref_texts = [val["ref"] for val in inputs.values()]
    ids = [Path(a).stem for a in audios]
    return prompts, audios, ref_texts, ids


def hf_inference(prompts, audios, model_path, batch_size=10):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation="flash_attention_2",
    ).to(device)
    generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    hf_results = []
    chunks = list(chunked(zip(audios, prompts), batch_size))
    for chunk in tqdm(chunks, desc="HF inference"):
        chunk_audios, chunk_prompts = zip(*chunk)
        inputs_batch = processor(text=list(chunk_prompts), audios=[sf_read(audio) for audio in chunk_audios], return_tensors="pt").to(device)
        # Only pass required arguments to model.generate
        generate_ids = model.generate(
            **inputs_batch,
            max_new_tokens=max_tokens,
            do_sample=False,
            # temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            # eos_token_id=stop_tokens_ids,
            pad_token_id=processor.tokenizer.pad_token_id,
            generation_config=generation_config,
        )

        generate_ids = generate_ids[:, inputs_batch["input_ids"].shape[1] :]
        responses = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        hf_results.extend(responses)
    return hf_results


def vllm_inference(prompts, audios, model_path, batch_size=100):
    client = VLLMClient(host=host)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt").input_ids.flatten().tolist()

    chunks = list(chunked(zip(audios, prompts), batch_size))
    vllm_results = []
    for chunk in tqdm(chunks, desc="vllm inference"):
        chunk_audios, chunk_prompts = zip(*chunk)
        # Only pass required arguments to client.generate
        chunk_responses = client.generate(
            list(chunk_prompts), audios=list(chunk_audios), n=n_gen, temperature=0, do_sample=False, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_tokens, stop_token_ids=stop_tokens_ids
        )
    vllm_results.extend(chunk_responses["texts"])
    return vllm_results


# %%
data_dir = home_dir / "data/librispeech_biasing/ref"
data_path = data_dir / "test-clean.biasing_100.jsonl"
prompts, audios, ref_texts, ids = prepare_dataset(data_path, bias_key=None)
# tgt_ids = ["5683-32866-0016","61-70968-0028"]
tgt_ids = ["5683-32866-0016","237-134500-0029","908-157963-0007","7176-92135-0006","8463-294825-0006","672-122797-0039","4970-29095-0009","5142-36586-0002","908-157963-0001","5142-33396-0009","7176-92135-0027","672-122797-0015","4446-2271-0013","5639-40744-0027","121-121726-0005","260-123440-0012","5142-36377-0000","3575-170457-0053","8224-274381-0010","1188-133604-0009","1995-1836-0010","8555-284449-0015","3570-5694-0018","7729-102255-0028","260-123288-0010","1320-122617-0036","237-134493-0017","1089-134686-0006","121-123859-0002","5105-28233-0009","7729-102255-0027","908-157963-0008","5683-32866-0026","8555-284447-0021","7176-92135-0045","5142-33396-0052","7729-102255-0042","8455-210777-0030","7127-75946-0003","1284-1180-0013","7176-92135-0020","2094-142345-0028","1320-122617-0013","4992-41806-0006","260-123288-0016","1995-1826-0008","8224-274384-0001","6930-76324-0007","8555-292519-0003","61-70970-0023"]
 
if tgt_ids is not None:
    idx = [ids.index(id) for id in tgt_ids]
    prompts = [prompts[i] for i in idx]
    audios = [audios[i] for i in idx]
    ref_texts = [ref_texts[i] for i in idx]
    ids = [ids[i] for i in idx]
print(f"Running on {len(ids)} utterances")
print("Running HuggingFace inference...")
hf_model_path = home_dir / "data/ckp/hf_models/phi4_mm_bias"
vllm_model_path = home_dir / "data/ckp/hf_models/phi4_mm_bias_merged"
hf_results = hf_inference(prompts, audios, hf_model_path, 32)
print("Running vLLM inference...")
vllm_results = vllm_inference(prompts, audios, vllm_model_path)
print("Comparing vLLM with HuggingFace outputs...")

for vllm_hyp, hf_hyp, ref, id in zip(vllm_results, hf_results, ref_texts, ids):

    vllm_wer, _, _ = compute_wers([{"id": id, "ref": ref, "hyp": vllm_hyp}])
    hf_wer, _, _ = compute_wers([{"id": id, "ref": ref, "hyp": hf_hyp}])

    if vllm_wer.get_wer() != hf_wer.get_wer():
        print("-" * 40)
        print(f"ID: {id}")
        print(f"VLLM WER: {vllm_wer.get_result_string()}")
        print(f"HF   WER: {hf_wer.get_result_string()}")
        print("*" * 40)
        print(f"      Reference: {ref}")
        print(f"VLLM Hypothesis: {vllm_hyp}")
        print(f"HF   Hypothesis: {hf_hyp}")


# %%
