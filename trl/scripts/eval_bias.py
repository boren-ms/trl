import torch
import wandb
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
import json
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from pathlib import Path
from trl.data_utils import sf_read
from trl.scripts.grpo_bias import init_model, create_dataset, make_parser, init_wandb
from trl.scripts.audio_metrics import compute_wers


class Evaluation:
    """Evaluation class for audio transcription biasing tasks."""

    def __init__(self, model_path, use_vllm=False, batch_size=8):
        self.accelerator = Accelerator()
        self.model_path = model_path
        self.batch_size = batch_size
        self.use_vllm = use_vllm

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.stop_tokens = ["<|end|>", self.processor.tokenizer.eos_token]
        self.stop_tokens_ids = self.processor.tokenizer(self.stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt").input_ids.flatten().tolist()

        self._prepare_model()
        if self.accelerator.is_main_process:
            init_wandb(
                job_name="test-eval",
                wandb_project="biasing_eval",
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
                limit_mm_per_prompt={"audio": 2},
            )
        else:
            self.model, _ = init_model(self.model_path)
            self.model.eval()
            self.model = self.accelerator.prepare(self.model)

    def generate(self, batch):
        """Generate outputs for a batch of audio files."""
        samples = [(sample["prompt"], sf_read(sample["audio_path"])) for sample in batch]

        if self.use_vllm:
            inputs = [{"prompt": prompt, "multi_modal_data": {"audio": [audio]}} for (prompt, audio) in samples]
            outputs = self.llm.generate(
                inputs,
                sampling_params=SamplingParams(temperature=0, max_tokens=512, n=1, stop_token_ids=self.stop_tokens_ids),
            )
        else:
            prompts, audios = zip(*samples)
            inputs = self.processor(text=prompts, audios=audios, return_tensors="pt").to(self.accelerator.device)
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
            )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            outputs = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return outputs

    def evaluate(self, dataset):
        assert dataset is not None, "Dataset must not be None"
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        dataloader = self.accelerator.prepare(dataloader)
        results = []
        for batch in dataloader:
            results += self.generate(batch)
        return results

    def measure(self, results, prefix=None):
        # Placeholder: implement your metric computation here
        # results = [{"hyp": ..., "ref": ..., "id": ...}, ...]
        wer, u_wer, b_wer = compute_wers(results)
        print("WER:", wer.get_result_string())  # noqa
        print("U-WER:", u_wer.get_result_string())  # noqa
        print("B-WER:", b_wer.get_result_string())  # noqa
        pfx = prefix + "/" if prefix else ""
        return {
            f"{pfx}WER": wer.get_wer(),
            f"{pfx}UWER": u_wer.get_wer(),
            f"{pfx}BWER": b_wer.get_wer(),
        }

    def evaluate_measures(self, dataset, name=None):
        """Evaluate the model on the dataset and compute metrics."""
        results = self.evaluate(dataset)
        all_results = gather_object(results)
        metrics = self.measure(all_results, prefix=name)
        if self.accelerator.is_main_process:
            wandb.log(metrics)
        return all_results, metrics

def main(args):
    """Main function to run the evaluation."""
    use_vllm = getattr(args, "use_vllm", False)
    evaluator = Evaluation(
        model_path=args.model_name_or_path,
        use_vllm=use_vllm,
        batch_size=getattr(args, "batch_size", 8),
    )
    output_dir = Path().home() / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = create_dataset(args.data_cfg)
    if not isinstance(datasets, torch.utils.data.Dataset):
        datasets = {"default": datasets}
    for name, dataset in datasets.items():
        print(f"Evaluating dataset: {name}")
        results, metrics = evaluator.evaluate_measures(dataset)
        if evaluator.accelerator.is_main_process:
            print(f"Metrics for {name}:")
            print(metrics)
            file_stem = f"{'vllm' if use_vllm else 'hf'}_{name}"
            result_file = output_dir / f"{file_stem}_results.json"
            metrics_file = output_dir / f"{file_stem}_metrics.json"
            with open(result_file, "w") as f:
                json.dump(results, f, indent=4)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to {metrics_file}")
            print(f"Results saved to {result_file}")
                

if __name__ == "__main__":
    parser = make_parser()
    script_args, _ = parser.parse_args_and_config()
    main(script_args)
