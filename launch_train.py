#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
from pathlib import Path
import fire
from ray_tool import (
    run_nodes,
    update_envs,
    prepare_env,
    prepare_data,
    release_gpus,
    prepare_local_output,
    sync_local_dir,
    init_ray,
    list_nodes,
    get_output_dirs,
    run_output_watcher
)
from launch_eval import evaluate_model

@ray.remote
def launch_training(config_file, output_dir):
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
    script_path = cur_dir / "trl/scripts/grpo_bias.py"
    cmd = [
        "accelerate",
        "launch",
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


def main(config_file, forced=False):
    """Launch the job on all nodes by preparing the environment and data."""
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
    run_nodes(launch_training, str(config_file), output_dir=str(output_dir))
    print("Training completed on all nodes.")

    print("Launching evaluation on all nodes")
    evaluate_model(local_model_dir=output_dir)
    print("Evaluation completed on all nodes.")

    watcher.flush.remote() 
    print("All tasks completed.")


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(main)
    # Example usage: python launch_job.py --config_file="path/to/config.yaml"
