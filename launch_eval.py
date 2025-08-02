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
    init_ray,
    list_nodes,
    get_local_path,
    get_remote_path,
    run_output_watcher,
    sync_local_dir,
    search_models,
    get_node_count,
    sorted_nodes,
)


@ray.remote
def launch_evaluation(model_path, config_file=None, nodes=None):
    """Launch evaluation using the specified YAML config file."""
    cur_dir = Path(__file__).parent
    os.chdir(cur_dir)
    print(f"Working Dir: {os.getcwd()}")

    config_file = config_file or "eval_conf/eval_baseline_hf.yaml"
    config_file = Path(config_file).absolute()
    update_envs(config_file)

    model_path = Path(model_path).absolute()

    print(f"Config file: {config_file}")
    print(f"Model path: {model_path}")
    assert model_path.exists(), f"Model {model_path} does not exist."

    head = 0 if nodes is None else nodes[0]
    rank = int(os.environ.get("RCALL_INSTANCE_INDEX", "0")) - head
    rank_size = get_node_count(nodes)
    num_gpu = int(os.environ.get("RCALL_NUM_GPU", "8"))
    job_name = os.environ.get("RCALL_JOB_NAME", None)
    assert job_name is not None, "RCALL_JOB_NAME must be set"
    main_process_ip = f"{job_name}-{head}"  # head node IP
    main_process_port = 12345
    script_path = cur_dir / "trl/scripts/eval_bias.py"
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
        "--model_path",
        str(model_path),
    ]

    rcall_logdir = os.environ.get("RCALL_LOGDIR", os.path.expanduser("~/logs"))
    os.makedirs(rcall_logdir, exist_ok=True)
    rank_log_file = os.path.join(rcall_logdir, f"{config_file.stem}_{model_path.stem}_{rank}.log")
    print(f"Logging to {rank_log_file}")
    with open(rank_log_file, "w") as logf:
        logf.write(f"Running {' '.join(cmd)}\n")
    # Optionally, printenv could be logged here

    with open(rank_log_file, "a") as logf:
        process = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def evaluate_model(remote_model_dir=None, local_model_dir=None, config=None, nodes=None):
    """Evaluate the model using the specified configuration."""

    if remote_model_dir:
        print(f"Evaluating {remote_model_dir}")
        local_model_dir = get_local_path(remote_model_dir)
        print("Preparing models from ", remote_model_dir)
        run_nodes(prepare_local_output, local_dir=local_model_dir, remote_dir=remote_model_dir, indexs=nodes)
    elif local_model_dir:
        print(f"Evaluating {local_model_dir}")
        remote_model_dir = get_remote_path(local_model_dir)
    else:
        raise ValueError("Either remote_model_dir or local_model_dir must be provided.")

    print("Syncing outputs from head to other nodes...")
    run_nodes(sync_local_dir, str(local_model_dir), nodes=nodes, indexs=nodes)

    print("Watching on ", local_model_dir)
    watcher = run_output_watcher(local_dir=local_model_dir, remote_dir=remote_model_dir, interval=120, sync_all=True, nodes=nodes)

    print(f"Evaluating {local_model_dir} with config {config}")
    run_nodes(launch_evaluation, local_model_dir, config, nodes=nodes, indexs=nodes)
    watcher.flush.remote()
    print("Evaluation completed on ", local_model_dir)


def main(model_path="", config=None, forced=False, nodes=None):
    """Launch the job on all nodes by preparing the environment and data."""
    init_ray()
    list_nodes()
    nodes = sorted_nodes(nodes)
    print(f"Using nodes: {nodes}")
    print(f"Search models: {model_path if model_path else 'default'}")
    model_paths = search_models(model_path)
    if not model_paths:
        print(f"No models found for {model_path}, existing evaluation.")
        return
    else:
        print(f"Found {len(model_paths)} models")
        for i, model_path in enumerate(model_paths):
            print(f"[{i}] Model: {model_path}")

    results = []
    print("Preparing environment on all nodes...")
    results += run_nodes(prepare_env, forced=forced, waiting=False, indexs=nodes)
    print("Preparing data on all nodes...")
    results += run_nodes(prepare_data, forced=forced, waiting=False, indexs=nodes)
    print("Releasing GPUs on all nodes...")
    results += run_nodes(release_gpus, waiting=False, indexs=nodes)
    ray.get(results)

    for model_path in model_paths:
        evaluate_model(remote_model_dir=model_path, config=config, nodes=nodes)


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(main)
    # Example usage: python launch_job.py --config_file="path/to/config.yaml"
