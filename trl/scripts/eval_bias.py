import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import soundfile as sf
import pandas as pd
import wandb

from trl.scripts.grpo_bias import init_model, is_master, create_dataset, make_parser


class Evaluation:
    def __init__(self, model_path, tsv_path, batch_size=8, n_gen=8):
        self.accelerator = Accelerator()
        self.model_path = model_path
        self.tsv_path = tsv_path
        self.batch_size = batch_size
        self.n_gen = n_gen

        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=512,
            max_num_seqs=batch_size,
            load_format="auto",
            limit_mm_per_prompt={"audio": 10},
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.stop_tokens = ["<|end|>", self.processor.tokenizer.eos_token]
        self.stop_tokens_ids = self.processor.tokenizer(
            self.stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
        ).input_ids.flatten().tolist()

        self.wav_paths = self.load_wav_path(tsv_path)
        self.prompts = self.build_prompts(self.wav_paths)
        # Use create_dataset instead of AudioDataset
        self.dataset = create_dataset(self.wav_paths, self.prompts)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)
        self.dataloader = self.accelerator.prepare(self.dataloader)

    def load_wav_path(self, tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", names=["id", "wav_paths", "msgs"])
        df["wav_path"] = df["wav_paths"].apply(lambda x: eval(x)[0])
        return df["wav_path"].tolist()

    def build_prompts(self, wav_paths):
        # Example: customize as needed
        words = ["cutlery", "utensils", "silverware", "TABLE", "CLOTHS", "Napkins", "Linen", "dining"]
        words_text = ", ".join([f"*{w}*" for w in words])
        text = "Transcribe the audio clip into text."
        text = f"{text} Please pay attention to following words: {words_text}."
        prompts = [f"<|user|><|audio_1|>{text}<|end|><|assistant|>" for _ in wav_paths]
        return prompts

    def evaluate(self):
        results = []
        for batch in self.dataloader:
            outputs = self.llm.generate(
                batch,
                sampling_params=SamplingParams(
                    temperature=1,
                    max_tokens=512,
                    n=self.n_gen,
                    stop_token_ids=self.stop_tokens_ids,
                ),
            )
            results.extend(outputs)
        return results

    def compute_metrics(self, results):
        # Placeholder: implement your metric computation here
        # Example: count number of outputs, or compute WER/CER if ground truth is available
        metrics = {
            "num_outputs": len(results),
            # Add more metrics as needed
        }
        return metrics

    def log_metrics(self, metrics):
        wandb.log(metrics)


def main(args):
    wandb.init(project="audio-bias-eval", config=vars(args))
    evaluator = Evaluation(
        model_path=args.model_path,
        tsv_path=args.tsv_path,
        batch_size=getattr(args, "batch_size", 8),
        n_gen=getattr(args, "n_gen", 8),
    )
    results = evaluator.evaluate()
    metrics = evaluator.compute_metrics(results)
    evaluator.log_metrics(metrics)
    if evaluator.accelerator.is_main_process:
        print("Evaluation Metrics:", metrics)


if __name__ == "__main__":
    parser = make_parser()
    script_args, _ = parser.parse_args_and_config()
    main(script_args)