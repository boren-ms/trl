import torch
import os
import sys
import pytz
import shortuuid
from pathlib import Path
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from transformers import AutoProcessor, GenerationConfig
import wandb
from trl import GRPOConfig, GRPOTrainer, TrlParser
import wandb
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import gather_object
import json
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from pathlib import Path
from trl.data_utils import sf_read
from trl.scripts.grpo_bias import init_model, create_dataset, make_parser, init_wandb
from trl.scripts.audio_metrics import compute_wers


@dataclass
class EvalArguments:
    """Script arguments for the  evaluation script."""

    job_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the script."},
    )
    project_name: Optional[str] = field(
        default="grpo_bias",
        metadata={"help": "Name of the project."},
    )

    eval_data: Optional[dict] = field(
        default=None,
        metadata={"help": "Evalution dataset config"},
    )
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model."},
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

class Evaluation:
    """Evaluation class for audio transcription biasing tasks."""

    def __init__(self, model_path, use_vllm=False, batch_size=8, output_dir=None, job_name=None, project_name=None, generation_config=None):
        self.accelerator = Accelerator()
        self.model_path = model_path
        self.batch_size = batch_size
        self.use_vllm = use_vllm
        self.output_dir = output_dir or model_path
        self.job_name = job_name
        self.project_name = project_name

        self.generation_config = GenerationConfig.from_pretrained(model_path, "generation_config.json")
        self.generation_config.update(**(generation_config or {}))
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self._prepare_model()
        if self.accelerator.is_main_process:
            init_wandb(
                job_name=self.job_name,
                wandb_project=self.project_name,
                config={
                    "model_path": model_path,
                    "use_vllm": use_vllm,
                    "batch_size": batch_size,
                },
            )

    def _prepare_model(self):
        """Prepare the model for evaluation."""
        if self.use_vllm:
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                max_model_len=512,
                max_num_seqs=self.batch_size,
                load_format="auto",
                limit_mm_per_prompt={"audio": 1},
            )
            config = hf2vllm_config(self.generation_config.to_dict())
            self.sampling_params = SamplingParams(**config)

        else:
            self.model, _ = init_model(self.model_path)
            self.model.eval()
            self.model = self.accelerator.prepare(self.model)

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
            generate_ids = self.model.generate(**inputs, generation_config=self.generation_config)
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            texts = outputs
        return texts

    def evaluate(self, dataset):
        assert dataset is not None, "Dataset must not be None"
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        dataloader = self.accelerator.prepare(dataloader)
        results = []
        for batch in dataloader:
            output = self.generate(batch)
            results += [{"hyp": hyp, "ref": ref, "id": id} for id, ref, hyp in zip(batch["id"], batch["text"], output)]
        return results

    def measure(self, results):
        # Placeholder: implement your metric computation here
        # results = [{"hyp": ..., "ref": ..., "id": ...}, ...]
        wer, u_wer, b_wer = compute_wers(results)
        print("WER:", wer.get_result_string())  # noqa
        print("U-WER:", u_wer.get_result_string())  # noqa
        print("B-WER:", b_wer.get_result_string())  # noqa

        return {
            f"WER": wer.get_wer(),
            f"UWER": u_wer.get_wer(),
            f"BWER": b_wer.get_wer(),
        }

    def evaluate_measures(self, dataset, name=None):
        """Evaluate the model on the dataset and compute metrics."""
        results = self.evaluate(dataset)
        all_results = gather_object(results)
        metrics = self.measure(all_results)
        return all_results, metrics

    def log_metrics_results(self, metrics, results, name=None):
        if not self.accelerator.is_main_process:
            return
        pfx = f"metric/{name}_" if name else "metric/"
        metrics = {f"{pfx}{k}": v for k, v in metrics.items()}
        wandb.log(metrics)
        print("Logging metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"{'vllm' if self.use_vllm else 'hf'}_{name}"
        result_file = output_dir / f"{file_stem}_results.json"
        metrics_file = output_dir / f"{file_stem}_metrics.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
        print(f"Results saved to {result_file}")


def make_parser(subparsers: argparse._SubParsersAction = None):
    """Create a parser for the evaluation script."""
    dataclass_types = EvalArguments
    if subparsers is not None:
        parser = subparsers.add_parser("eval", help="Run the evaluation script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


def main(args):
    """Main function to run the evaluation."""

    eval_data = args.eval_data
    evaluator = Evaluation(
        model_path=args.model_name_or_path,
        output_dir=args.output_dir,
        use_vllm=args.use_vllm,
        batch_size=args.batch_size,
        job_name=args.job_name,
        project_name=args.project_name,
        generation_config=args.generation_config,
    )
    datasets = create_dataset(eval_data)
    if not isinstance(datasets, dict):
        datasets = {"default": datasets}
    for name, dataset in datasets.items():
        print(f"Evaluating dataset: {name}")
        results, metrics = evaluator.evaluate_measures(dataset)
        evaluator.log_metrics_results(metrics, results, name=name)
        print("*" * 20)
    print("Evaluation completed.")



if __name__ == "__main__":
    parser = make_parser()
    eval_args = parser.parse_args_and_config()
    main(eval_args[0])
