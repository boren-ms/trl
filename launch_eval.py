#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
from pathlib import Path
import fire
from ray_tool import (
    ORNG_USER,
    run_nodes,
    update_envs,
    prepare_env,
    prepare_data,
    release_gpus,
    prepare_local_output,
    init_ray,
    list_nodes,
    get_output_dirs,
    run_output_watcher,
    sync_folder,
    is_valid_model_path,
)


@ray.remote
def launch_evaluation(model_path, config_file=None):
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

    rank = int(os.environ.get("RCALL_INSTANCE_INDEX", "0"))
    rank_size = int(os.environ.get("RCALL_INSTANCE_COUNT", "1"))
    num_gpu = int(os.environ.get("RCALL_NUM_GPU", "8"))
    job_name = os.environ.get("RCALL_JOB_NAME", None)
    assert job_name is not None, "RCALL_JOB_NAME must be set"
    main_process_ip = f"{job_name}-0"  # head node IP
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



def main(model_name, config_file=None, forced=False):
    """Launch the job on all nodes by preparing the environment and data."""
    init_ray()
    list_nodes()

    print(f"Model name: {model_name}")
    model_dir, remote_model_dir = get_output_dirs(Path(model_name).stem)

    results = []
    print("Preparing environment on all nodes...")
    results += run_nodes(prepare_env, forced=forced, waiting=False)

    print("Preparing data on all nodes...")
    results += run_nodes(prepare_data, forced=forced, waiting=False)

    print("Releasing GPUs on all nodes...")
    results += run_nodes(release_gpus, waiting=False)

    print("Preparing local models")
    results += run_nodes(prepare_local_output, local_dir=model_dir, remote_dir=remote_model_dir, waiting=False)

    # Ensure all tasks are completed before proceeding
    ray.get(results)
    print("Syncing outputs from head to other nodes...")
    run_nodes(sync_folder, str(model_dir))
    
    if not is_valid_model_path(model_dir):  # for baseline models
        print(f"Model [{model_name}] is not valid in {model_dir}, switching to prepared data checkpoints folder")
        rel_path = f"data/ckp/hf_models/{model_name}"
        model_dir = Path.home() / rel_path
        remote_model_dir = f"{ORNG_USER.home_path}/{rel_path}"

    if not is_valid_model_path(model_dir):
        print(f"Model [{model_name}] is not valid in {model_dir}")
        print(f"Exiting evaluation, please double check model [{model_name}]")
        return
    
    print("Starting output watcher on head node...")
    run_output_watcher(local_dir=model_dir, remote_dir=remote_model_dir, interval=600)
    print(f"Evaluating {model_dir} ")
    run_nodes(launch_evaluation, model_dir, config_file)
    print("All tasks completed, stopping watcher.")


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(main)
    # Example usage: python launch_job.py --config_file="path/to/config.yaml"
