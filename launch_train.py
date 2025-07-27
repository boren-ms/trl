#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial
import subprocess
import ray
import os
from pathlib import Path
import fire
from ray_tool import run_nodes, update_envs, prepare_env, prepare_data, release_gpus, prepare_local_output, sync_local_dir, init_ray, list_nodes, get_output_dirs, run_output_watcher
from launch_eval import evaluate_model


@ray.remote
def launch_training(script_path, config_file, output_dir, acc_config=None):
    """Launch training using the specified YAML config file."""
    config_file = Path(config_file).absolute()
    update_envs(config_file)

    cur_dir = Path(__file__).parent
    os.chdir(cur_dir)
    print(f"Working Dir: {os.getcwd()}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Using config file: {config_file}")
    print(f"Output directory: {output_dir}")

    rank = int(os.environ.get("RCALL_INSTANCE_INDEX", "0"))
    rank_size = int(os.environ.get("RCALL_INSTANCE_COUNT", "1"))
    num_gpu = int(os.environ.get("RCALL_NUM_GPU", "8"))
    job_name = os.environ.get("RCALL_JOB_NAME", None)
    assert job_name is not None, "RCALL_JOB_NAME must be set"
    main_process_ip = f"{job_name}-0"  # head node IP
    main_process_port = 12345
    acc_args = ["--config_file", str(acc_config)] if acc_config else []
    cmd = [
        "accelerate",
        "launch",
        *acc_args,
        "--num_processes",
        str(num_gpu * rank_size),
        "--num_machines",
        str(rank_size),
        "--machine_rank",
        str(rank),
        "--main_process_ip",
        str(main_process_ip),
        "--main_process_port",
        str(main_process_port),
        str(script_path),
        "--config",
        str(config_file),
        "--output-dir",
        str(output_dir),
    ]

    rcall_logdir = os.environ.get("RCALL_LOGDIR", os.path.expanduser("~/logs"))
    os.makedirs(rcall_logdir, exist_ok=True)
    rank_log_file = os.path.join(rcall_logdir, f"{config_file.stem}_rank_{rank}.log")
    print(f"Logging to {rank_log_file}")
    with open(rank_log_file, "w") as logf:
        logf.write(f"Running {' '.join(cmd)}\n")
    # Optionally, printenv could be logged here

    with open(rank_log_file, "a") as logf:
        process = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def get_task_script(task=None, config_file=None):
    """Return the script path for the given task."""
    assert task or config_file, "Either task or config_file must be provided"
    cur_dir = Path(__file__).parent
    tasks = {
        "grpo": cur_dir / "trl/scripts/grpo_bias.py",
        "dpo": cur_dir / "trl/scripts/online_dpo_bias.py",
        "online_dpo": cur_dir / "trl/scripts/online_dpo_bias.py",  # add alias
    }

    if not task and config_file:
        name_parts = Path(config_file).stem.split("_")
        task = next((t for t in tasks if t in name_parts), None)
    assert task, "Task must be specified or inferred from config_file"
    script_path = tasks[task]
    assert script_path.exists(), f"Script {script_path} does not exist."
    return script_path


def get_acc_config(name=None):
    """Return the accelerate config file path for the given name."""
    cwd = Path(__file__).parent
    name_dict = {
        "zero1": cwd / "examples/accelerate_configs/deepspeed_zero1.yaml",
        "zero2": cwd / "examples/accelerate_configs/deepspeed_zero2.yaml",
        "zero3": cwd / "examples/accelerate_configs/deepspeed_zero3.yaml",
        "fsdp1": cwd / "examples/accelerate_configs/fsdp1.yaml",
        "fsdp2": cwd / "examples/accelerate_configs/fsdp2.yaml",
    }
    return name_dict.get(name, None)


def main(config_file, task=None, forced=False, acc="zero2"):
    """Launch the job on all nodes by preparing the environment and data."""
    script_path = get_task_script(task, config_file)
    print(f"Using script: {script_path}")
    init_ray()
    list_nodes()

    config_file = Path(config_file).absolute()
    print(f"Using config file: {config_file}")
    output_dir, remote_output_dir = get_output_dirs(config_file.stem)

    results = []
    print("Preparing environment on all nodes...")
    results += run_nodes(prepare_env, forced=forced, waiting=False)

    print("Preparing data on all nodes...")
    results += run_nodes(prepare_data, forced=forced, waiting=False)

    print("Releasing GPUs on all nodes...")
    results += run_nodes(release_gpus, waiting=False)

    print("Preparing output on all nodes...")
    results += run_nodes(prepare_local_output, local_dir=output_dir, remote_dir=remote_output_dir, waiting=False)

    # Ensure all tasks are completed before proceeding
    ray.get(results)

    print("Syncing outputs from head to other nodes...")
    run_nodes(sync_local_dir, str(output_dir))

    print("Starting output watcher on head node...")
    watcher = run_output_watcher(local_dir=output_dir, remote_dir=remote_output_dir, interval=600)

    print(f"Launching training with {config_file}...")
    run_nodes(launch_training, str(script_path), str(config_file), output_dir=str(output_dir), acc_config=get_acc_config(acc))
    print("Training completed on all nodes.")

    print("Launching evaluation on all nodes")
    evaluate_model(local_model_dir=output_dir)
    print("Evaluation completed on all nodes.")

    watcher.flush.remote()
    print("All tasks completed.")


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(main)
    # Example usage: python launch_job.py  --config_file="path/to/config.yaml"
