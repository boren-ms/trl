# %%
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from tqdm import tqdm
from more_itertools import chunked
from trl.scripts.audio_dataset import ls_bias_dataset
from trl.scripts.audio_metrics import compute_wers
from trl.data_utils import sf_read
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from itertools import zip_longest

# ----------------- Config -----------------
home_dir = Path.home()

REMOTE_DIR = "az://orngscuscresco/data/boren/data"

n_gen = 1
max_tokens = 512
temperature = 0.0
top_p = 1.0
top_k = -1
min_p = 0.0
repetition_penalty = 1.0
device = "cuda:7"


def prepare_dataset(data_path, bias_key="distractors", num_egs=None, idx_list=None):
    ds = ls_bias_dataset(str(data_path), bias_key=bias_key, data_dir=REMOTE_DIR)
    if num_egs is not None:
        ds = ds.select(range(num_egs))
    prompts = []
    audio_paths = []
    texts = []
    ids = []
    for sample in ds:
        idx = Path(sample["audio_path"]).stem
        if idx_list is not None and idx not in idx_list:
            continue
        prompts.append(sample["prompt"])
        audio_paths.append(sample["audio_path"])
        texts.append(sample["text"])
        ids.append(idx)

    return prompts, audio_paths, texts, ids


def hf_inference(model, processor, prompts, audios, batch_size=10, generation_config=None):
    """Run inference using HuggingFace Transformers."""
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


def vllm_inference(llm, prompts, audios, batch_size=100, sampling_params=None, speech_lora_path=None):
    """Run inference using vLLM."""
    chunks = list(chunked(zip(audios, prompts), batch_size))
    for chunk in tqdm(chunks, desc="vllm inference"):
        chunk_audios, chunk_prompts = zip(*chunk)
        inputs = [{"prompt": prompt, "multi_modal_data": {"audio": [sf_read(audio_path)]}} for prompt, audio_path in zip(chunk_prompts, chunk_audios)]
        lora_request = [LoRARequest("speech", 1, str(speech_lora_path))] * len(inputs) if speech_lora_path else None
        outputs = llm.generate(inputs, sampling_params=sampling_params, lora_request=lora_request)
        texts = [output.outputs[0].text for output in outputs]
    return texts


# %%

# %%
data_dir = home_dir / "data/librispeech_biasing/ref"
data_path = data_dir / "test-clean.biasing_100.jsonl"
tgt_ids = ["5683-32866-0016", "237-134500-0029"]
# prompts, audios, texts, ids = prepare_dataset(data_path, bias_key=None, idx_list=tgt_ids)
prompts, audios, texts, ids = prepare_dataset(data_path, bias_key=None, num_egs=10)
print(f"Running on {len(ids)} utterances")
model_path = home_dir / "data/ckp/hf_models/Phi-4-multimodal-instruct"
# model_path = home_dir / "data/ckp/hf_models/phi4_mm_bias"
hf_results = []
# %%
# print("Running HuggingFace inference...", model_path)
# generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
# processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype="auto", _attn_implementation="flash_attention_2").to(device)
# hf_results = hf_inference(model, processor, prompts, audios, 32, generation_config=generation_config)

# %%

os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
print(f"Running vLLM inference...: {model_path}")
llm = LLM(
    model=str(model_path),
    trust_remote_code=True,
    max_model_len=10240,
    max_num_seqs=2,
    enable_lora=True,
    limit_mm_per_prompt={"audio": 2},
    max_lora_rank=320,
    # distributed_executor_backend="external_launcher",
    # seed=0,
)
generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
speech_lora_path = model_path / "speech-lora"
sampling_params = SamplingParams(
    n=n_gen,
    temperature=temperature,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
    max_tokens=max_tokens,
    stop_token_ids=generation_config.eos_token_id,
)
# speech_lora_path = None
vllm_results = vllm_inference(llm, prompts, audios, sampling_params=sampling_params, speech_lora_path=speech_lora_path)
del llm  # release GPUs
# %%
print("Comparing vLLM with HuggingFace outputs...")
for vllm_hyp, hf_hyp, ref, id in zip_longest(vllm_results, hf_results, texts, ids):
    vllm_wer, _, _ = compute_wers([{"id": id, "ref": ref, "hyp": vllm_hyp}])
    if hf_hyp is not None:
        hf_wer, _, _ = compute_wers([{"id": id, "ref": ref, "hyp": hf_hyp}])

    print("-" * 40)
    print(f"ID: {id}")
    print(f"VLLM WER: {vllm_wer.get_result_string()}")
    if hf_hyp is not None:
        print(f"HF   WER: {hf_wer.get_result_string()}")
    print("*" * 40)
    print(f"      Reference: {ref}")
    print(f"VLLM Hypothesis: {vllm_hyp}")
    if hf_hyp is not None:
        print(f"HF   Hypothesis: {hf_hyp}")


# %%
