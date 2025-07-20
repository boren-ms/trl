import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
from transformers import GenerationConfig
from trl import TrlParser
import wandb
from more_itertools import unique_everseen
from torch.utils.data import DataLoader
from accelerate import Accelerator, find_executable_batch_size
from accelerate.utils import gather_object
import json
from vllm import LLM, SamplingParams
from pathlib import Path
from trl.data_utils import sf_read, find_chkps, chkp_index
from trl.scripts.grpo_bias import init_model, create_dataset, make_parser, init_wandb
from trl.scripts.audio_metrics import compute_wers


@dataclass
class EvalArguments:
    """Script arguments for the  evaluation script."""

    job_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the job."},
    )
    eval_data: Optional[dict] = field(
        default=None,
        metadata={"help": "Evaluation dataset config"},
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
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
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for evaluation results."},
    )
    generation_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Generation parameters for the model."},
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


class Evaluation:
    """Evaluation class for audio transcription biasing tasks."""

    def __init__(self, model_path, use_vllm=False, batch_size=8, output_dir=None, job_name=None, wandb_dir=None, generation_config=None):
        self.accelerator = Accelerator()
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.use_vllm = use_vllm
        self.output_dir = output_dir or model_path
        self.wandb_dir = wandb_dir or self.output_dir
        self.job_name = job_name

        self.generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
        self.generation_config.update(**(generation_config or {}))
        hack_package(self.model_path, True)
        self._prepare_model()
        if self.is_main:
            init_wandb(job_name=self.job_name, config={"model_path": model_path, "use_vllm": use_vllm, "batch_size": batch_size}, output_dir=self.wandb_dir)

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
            max_all_tokens = 1024*6
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                max_model_len=max_all_tokens,  # the max token processed by vLLM including both input and output
                distributed_executor_backend="external_launcher",
                seed=self.rank,
                max_num_seqs=self.batch_size,
                load_format="auto",
                limit_mm_per_prompt={"audio": 1},
                max_num_batched_tokens=max_all_tokens*2,
            )
            config = hf2vllm_config(self.generation_config.to_dict())
            self.sampling_params = SamplingParams(**config)
        else:
            model, self.processor = init_model(self.model_path)
            self.model = self.accelerator.prepare(model).eval()

    def generate(self, batch):
        """Generate outputs for a batch of audio files."""
        texts = []
        if self.use_vllm:
            inputs = [{"prompt": prompt, "multi_modal_data": {"audio": [sf_read(audio_path)]}} for prompt, audio_path in zip(batch["prompt"], batch["audio_path"])]
            outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
            texts = [output.outputs[0].text for output in outputs]
        else:
            audios = [sf_read(audio_path) for audio_path in batch["audio_path"]]
            inputs = self.processor(text=batch["prompt"], audios=audios, return_tensors="pt").to(self.accelerator.device)
            model = self.accelerator.unwrap_model(self.model)
            generate_ids = model.generate(**inputs, generation_config=self.generation_config)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            texts = outputs
        return texts

    def evaluate(self, dataset):
        assert dataset is not None, "Dataset must not be None"

        @find_executable_batch_size(starting_batch_size=self.batch_size)
        def auto_eval(batch_size):
            self.rank_log("Evaluating batch size:", batch_size, all=True)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            dataloader = self.accelerator.prepare(dataloader)
            results = []
            for batch in tqdm(dataloader, desc="Evaluating batches", disable=not self.is_main):
                output = self.generate(batch)
                results += [{"hyp": hyp, "ref": ref, "id": id, "prompt": prompt} for id, ref, prompt, hyp in zip(batch["id"], batch["text"], batch["prompt"], output)]
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
    evaluator = Evaluation(
        model_path=model_path,
        use_vllm=kwargs.get("use_vllm", False),
        batch_size=kwargs.get("batch_size", 8),
        output_dir=kwargs.get("output_dir", None),
        job_name=kwargs.get("job_name", None),
        wandb_dir=kwargs.get("wandb_dir", None),
        generation_config=kwargs.get("generation_config", None),
    )
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
