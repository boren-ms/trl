import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm
from transformers import GenerationConfig
from trl import TrlParser
import wandb
from more_itertools import unique_everseen
from itertools import zip_longest
from torch.utils.data import DataLoader, Sampler
from accelerate import Accelerator, find_executable_batch_size
from accelerate.utils import gather_object
from collections import defaultdict
import json
from vllm import LLM, SamplingParams
from pathlib import Path
import blobfile as bf
from trl.data_utils import load_audio, find_chkps, chkp_index
from trl.scripts.grpo_bias import init_model, create_dataset, make_parser, WandbHelper
from trl.scripts.audio_metrics import compute_wers
from trl.trainer.utils import move_model_to_vllm
from trl.scripts.chunk.svad import SVadChunker


@dataclass
class EvalArguments:
    """Script arguments for the  evaluation script."""

    job_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the job."},
    )
    new_run: bool = field(default=False, metadata={"help": "whether to skip run info from checkpoint"})
    eval_data: Optional[dict] = field(
        default=None,
        metadata={"help": "Evaluation dataset config"},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
    )
    model_with_lora: bool = field(
        default=True,
        metadata={"help": "Whether the model is with LoRA adapters."},
    )
    checkpoints: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Checkpoint indices to evaluate."},
    )
    batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Batch size for evaluation."},
    )
    use_vllm: Optional[bool] = field(
        default=True,
        metadata={"help": "Use vLLM for evaluation."},
    )
    vllm_max_length: int = field(default=1024 * 10, metadata={"help": "max model length for vllm inference"})
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for evaluation results."},
    )
    generation_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Generation parameters for the model."},
    )
    max_chunk_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum segment length for audio chunking in seconds."},
    )
    with_history: bool = field(
        default=False,
        metadata={"help": "whether to use history for next chunk decoding."},
    )


def hf2vllm_config(hf_config):
    """Convert a HuggingFace GenerationConfig or dict to vLLM sampling parameters."""
    mapping = {
        "max_new_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "num_return_sequences": "n",
        "eos_token_id": "stop_token_ids",
    }
    vllm_params = {vllm_key: hf_config[hf_key] for hf_key, vllm_key in mapping.items() if hf_key in hf_config and hf_config[hf_key] is not None}
    if "do_sample" in hf_config and not hf_config["do_sample"]:
        vllm_params["temperature"] = 0.0
    vllm_params["n"] = vllm_params.get("n", 1)
    return vllm_params


def hack_package(package_path, replace=False):
    """hack the processor config file to remove unnecessary keys and rename some keys."""
    if not Path(package_path).exists():
        print(f"Model path {package_path} is not local path, skip to hack.")
        return
    chat_temp_file = Path(package_path) / "chat_template.json"
    chat_temp_file.unlink(missing_ok=True)

    preprocessor_conf = Path(package_path) / "preprocessor_config.json"
    assert preprocessor_conf.exists(), f"{preprocessor_conf} does not exist."
    print(f"Loading PreProcessor Config: {preprocessor_conf}")
    data = json.load(open(preprocessor_conf, "r", encoding="utf-8"))
    print("Original PreProcessor Config:")
    print(json.dumps(data, indent=4))
    key_mapping = {
        "auto_map": None,
        "image_processor_type": None,
        "processor_class": None,
        "feature_extractor_type": None,
        "audio_compression_rate": "compression_rate",
        "audio_downsample_rate": "qformer_compression_rate",
        "audio_feat_stride": "feat_stride",
        "dynamic_hd": None,
    }

    if set(data.keys()) == set(key_mapping.keys()):
        print("All keys match the config, skip to hack.")
        return

    # Filter out keys with None values from key_mapping
    new_data = {}
    for key, old_key in key_mapping.items():
        if key in data:
            new_data[key] = data[key]
        elif old_key is not None and old_key in data:
            new_data[key] = data[old_key]
        else:
            raise KeyError(f"Key '{key}[{old_key}]' not found.")
    print("New PreProcessor Config:")
    print(json.dumps(new_data, indent=4))
    if not replace:
        return
    with open(preprocessor_conf, "wt") as f:
        json.dump(new_data, f, indent=4)
    print(f"Updated config file: {preprocessor_conf}")


def file_size(file_path):
    if not bf.exists(file_path):
        return -1
    return bf.stat(file_path).size


class LengthBatchSampler(Sampler[List[int]]):
    """Batch sampler that groups data by length."""

    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Yield batches of indices grouped by length."""
        # sort by the transcription length
        sizes = torch.tensor([file_size(x.get("audio_path", "")) for x in self.dataset])
        for batch in torch.chunk(torch.argsort(sizes), len(self)):
            yield batch.tolist()


class Evaluation:
    """Evaluation class for audio transcription biasing tasks."""

    def __init__(self, model_path, use_vllm=False, vllm_max_length=1024 * 10, batch_size=8, output_dir=None, job_name=None, wandb_dir=None, generation_config=None, new_run=False, **kwargs):
        self.accelerator = Accelerator()
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.use_vllm = use_vllm
        self.output_dir = output_dir or model_path
        self.wandb_dir = wandb_dir or self.output_dir
        self.job_name = job_name
        self.vllm_max_length = vllm_max_length
        self.max_chunk_len = kwargs.get("max_chunk_len", None)
        self.with_history = kwargs.get("with_history", False)
        self.model_with_lora = kwargs.get("model_with_lora", True)
        self.chk_idx_name = "_chk_idx_"

        self.chunker = SVadChunker(max_len_sec=self.max_chunk_len) if self.max_chunk_len else None
        WandbHelper(run_name=self.job_name, work_dir=self.wandb_dir, new_run=new_run).init(main_only=True)

        self.generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
        self.generation_config.update(**(generation_config or {}))
        hack_package(self.model_path, True)
        self._prepare_model()

    @property
    def rank(self):
        """Get the rank of the current process."""
        return self.accelerator.process_index

    @property
    def is_main(self):
        """Check if the current process is the main process."""
        return self.accelerator.is_main_process

    def rank_log(self, *args, all=False):
        """Log a message with the current rank."""
        if not all and not self.is_main:
            return
        print(f"[{self.rank}]", *args)

    def _prepare_model(self):
        """Prepare the model for evaluation."""
        if self.use_vllm:
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                max_model_len=self.vllm_max_length,  # the max token processed by vLLM including both input and output
                distributed_executor_backend="external_launcher",
                seed=self.rank,
                max_num_seqs=self.batch_size,
                load_format="auto",
                limit_mm_per_prompt={"audio": 1},
            )
            if self.model_with_lora:
                # if model with LoRA, we need merge lora back to normal Linear, and then load the model with LoRA adapters
                model, _ = init_model(self.model_path)
                move_model_to_vllm(model, self.llm)
                del model  # no need any more.
            config = hf2vllm_config(self.generation_config.to_dict())
            self.sampling_params = SamplingParams(**config)
        else:
            model, self.processor = init_model(self.model_path)
            self.model = self.accelerator.prepare(model).eval()

    def _generate_single(self, examples):
        """Generate outputs for a batch of audio files."""

        def load_prompt(x):
            """load prompt with/without context"""

            prompt = x["prompt"]
            if not self.with_history:
                return prompt
            contexts = x.get("history", [])
            if not contexts:
                return prompt
            HISTORY_PROMPT = "Please consider the following context:"
            TAIL_STR = "<|end|><|assistant|>"
            prefix = prompt.replace(TAIL_STR, "").strip()
            return f"{prefix} {HISTORY_PROMPT} {contexts[-1]} {TAIL_STR}"

        if self.use_vllm:
            inputs = [{"prompt": load_prompt(x), "multi_modal_data": {"audio": [load_audio(x)]}} for x in examples]
            outputs = self.llm.generate(inputs, sampling_params=self.sampling_params, use_tqdm=False)
            texts = [output.outputs[0].text for output in outputs]
        else:
            audios = [load_audio(x) for x in examples]
            prompts = [load_prompt(x) for x in examples]
            inputs = self.processor(text=prompts, audios=audios, return_tensors="pt").to(self.accelerator.device)
            model = self.accelerator.unwrap_model(self.model)
            generate_ids = model.generate(**inputs, generation_config=self.generation_config)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            texts = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return texts

    def _chunk_audio(self, data, sr):
        """Chunk audio data into segments."""
        n_sec = len(data) / sr
        if self.chunker is None or n_sec <= self.max_chunk_len:
            return [(0, n_sec)]
        return self.chunker.chunk(data, sr)

    def _chunk_batch(self, examples):
        """Chunk a batch of examples into segments."""
        batches = []
        for example in tqdm(examples, desc="Chunking examples", disable=not self.is_main):
            sub_examples = []
            data, sr = load_audio(example)
            for i, (s, e) in enumerate(self._chunk_audio(data, sr)):
                chk_egs = example.copy()
                chk_egs["audio"] = data[int(s * sr) : int(e * sr)]
                chk_egs["sr"] = sr
                chk_egs[self.chk_idx_name] = i
                sub_examples.append(chk_egs)
            batches.append(sub_examples)
        batches = list(zip_longest(*batches, fillvalue=None))
        return batches

    def generate(self, examples):
        """Generate outputs for a batch of examples."""
        IDX_KEY = "_egs_idx_"
        examples = [{**x, IDX_KEY: i} for i, x in enumerate(examples)]
        chunks = self._chunk_batch(examples)
        if not self.with_history:  # flatten the chunks if not using history
            chunks = [chunk for batch in chunks for chunk in batch if chunk is not None]
            chunks = [chunks[x : x + self.batch_size] for x in range(0, len(chunks), self.batch_size)]

        output_dict = defaultdict(list)
        for chk_inputs in tqdm(chunks, desc="Evaluating chunks", disable=not self.is_main):
            chk_inputs = [{**x, "history": output_dict[x[IDX_KEY]]} for x in chk_inputs if x is not None]
            chk_outputs = self._generate_single(chk_inputs)
            for inp, oup in zip(chk_inputs, chk_outputs):
                output_dict[inp[IDX_KEY]].append((inp[self.chk_idx_name], oup))
        all_outputs = [" ".join([x[1] for x in sorted(output_dict[k], key=lambda x: x[0])]) for k in sorted(output_dict.keys())]
        return all_outputs

    def evaluate(self, dataset):
        assert dataset is not None, "Dataset must not be None"

        @find_executable_batch_size(starting_batch_size=self.batch_size)
        def auto_eval(batch_size):
            self.rank_log("Evaluating batch size:", batch_size, all=True)
            dl_kwargs = {
                "collate_fn": lambda x: x,
                "batch_size": batch_size,
                # "num_workers": 2,
                # "prefetch_factor": 2,
            }
            # batch_sampler = LengthBatchSampler(dataset, batch_size)
            # dataloader = self.accelerator.prepare(DataLoader(dataset, batch_sampler=batch_sampler, **dl_kwargs))
            dataloader = self.accelerator.prepare(DataLoader(dataset, **dl_kwargs))
            results = []
            keys = ["hyp", "ref", "audio_path", "id", "WER", "UWER", "BWER", "keywords", "Transcription"]
            for inputs in tqdm(dataloader, desc="Evaluating batches", disable=not self.is_main):
                outputs = self.generate(inputs)
                for x, hyp in zip(inputs, outputs):
                    x["hyp"] = hyp
                    x["ref"] = x["text"]
                    x.update(self.measure([x]))
                    results.append({k: x.get(k, None) for k in keys})
            return results

        return auto_eval()

    def measure(self, results):
        # Placeholder: implement your metric computation here
        # results = [{"hyp": ..., "ref": ..., "id": ...}, ...]
        wer, u_wer, b_wer = compute_wers(results)
        return {
            f"WER": wer.get_wer(),
            f"UWER": u_wer.get_wer(),
            f"BWER": b_wer.get_wer(),
        }

    def evaluate_measures(self, dataset):
        """Evaluate the model on the dataset and compute metrics."""
        results = self.evaluate(dataset)
        all_results = gather_object(results)
        unique_results = list(unique_everseen(all_results, key=lambda x: x["id"]))  # remove duplicates from multiple ranks
        metrics = self.measure(unique_results)
        return all_results, metrics

    def log_metrics_results(self, metrics, results, name=None):
        if not self.is_main:
            return
        pfx = "metric_{}/".format("vllm" if self.use_vllm else "hf")
        if name:
            pfx += f"{name}_"

        metrics = {k if "/" in k else f"{pfx}{k}": v for k, v in metrics.items()}  # skip prefix for keys with slashes
        self.rank_log("Logging metrics:")
        for key, value in metrics.items():
            self.rank_log(f"{key}: {value}")

        wandb.log(metrics)

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"{'vllm' if self.use_vllm else 'hf'}_{name}"
        result_file = output_dir / f"{file_stem}_results.json"
        metrics_file = output_dir / f"{file_stem}_metrics.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        self.rank_log(f"Metrics saved to {metrics_file}")
        self.rank_log(f"Results saved to {result_file}")

    def evaluate_all(self, datasets, step=0):
        """Evaluate the model on all datasets."""
        if not isinstance(datasets, dict):
            datasets = {"default": datasets}

        for name, dataset in tqdm(datasets.items(), "Evaluating datasets", disable=not self.is_main):
            self.rank_log(f"Evaluating {self.model_path} @ {name}")
            results, metrics = self.evaluate_measures(dataset)
            metrics.update({"train/global_step": step})
            self.log_metrics_results(metrics, results, name=name)


def find_models(model_path, checkpoints=None):
    """Get a list of checkpoint directories in the model path."""
    chkps = find_chkps(model_path, checkpoints)
    if checkpoints is None:
        return [chkps[0] if chkps else model_path]  # latest or current folder
    return chkps


def evaluate_model(model_path, datasets, **kwargs):
    """Evaluate a model on the given evaluation data."""
    evaluator = Evaluation(model_path=model_path, **kwargs)
    evaluator.evaluate_all(datasets, step=chkp_index(Path(model_path).name, 0))
    del evaluator  # Clean up the evaluator to free resources


def main(args):
    """Main function to run the evaluation."""
    model_path = Path(args.model_path)
    job_name = args.job_name or model_path.parent.stem if model_path.stem.startswith("checkpoint-") else model_path.stem
    model_paths = find_models(args.model_path, args.checkpoints)
    datasets = create_dataset(args.eval_data)
    kwargs = {k: v for k, v in vars(args).items() if k not in ["model_path", "eval_data", "checkpoints", "job_name"]}

    for model_path in model_paths:
        evaluate_model(model_path, datasets, wandb_dir=args.model_path, job_name=job_name, **kwargs)


def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create a parser for the evaluation script."""
    dataclass_types = EvalArguments
    if subparsers is not None:
        parser = subparsers.add_parser("eval", help="Run the evaluation script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    eval_args = parser.parse_args_and_config()
    main(eval_args[0])
