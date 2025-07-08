#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
import importlib
from pathlib import Path
import fire

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    

def run_nodes(fun, *args, **kwargs):
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]

    # Launch one task per node, each pinned to a specific node
    results = []
    for node_ip in node_ips:
        # Use custom resource label to ensure the function runs on this node
        # Each node has a resource label 'node:<ip>'
        node_label = f"node:{node_ip}"
        result = fun.options(resources={node_label: 0.01}).remote(*args, **kwargs)
        results.append(result)
    return ray.get(results)


REGION_STORAGES = {
    "southcentralus": "orngscuscresco",
    "westus2": "orngwus2cresco",
    "uksouth": "orngukscresco",
}
def get_region_storage():
    """ Get the storage path based on the region of the Kubernetes cluster."""
    rcall_kube_cluster = os.environ.get("RCALL_KUBE_CLUSTER", "")
    cluster_region = rcall_kube_cluster.split("-")[1] if "-" in rcall_kube_cluster else ""
    data_storage = REGION_STORAGES.get(cluster_region, "orngscuscresco")
    user = os.environ.get("RCALL_USER", "boren")
    return f"az://{data_storage}/data/{user}/data"

#%%
@ray.remote
def prepare_environment():
    """ Prepare the environment on each node by installing necessary packages."""
    try:
        trl_mod = importlib.import_module("trl")
        trl_path = trl_mod.__file__
        if trl_path.startswith("/root/code/trl"):
            print("trl is installed from /root/code/trl")
            print("Environment is already prepared, skipping reinstallation.")
            return 
    except ImportError as e:
        print("Could not determine trl installation path:", e)
    hostname = os.uname().nodename
    print(f"Preparing environment on node: {hostname}")
    run("pip uninstall -y torch torchvision torchaudio transformers flash-attn vllm trl")
    run("uv pip install --system torch==2.6.0 ray==2.36.1 torchvision torchaudio transformers==4.51.3 vllm trl peft tensorboardX blobfile soundfile more-itertools whisper_normalizer fire")
    run("pip install torch==2.6.0 flash-attn")
    run("pip uninstall -y trl")
    print("Environment preparation completed.")

@ray.remote
def prepare_data():
    """ Prepare data on each node by syncing from the remote storage."""
    hostname = os.uname().nodename
    print(f"Preparing data on node: {hostname}")
    local_dir = Path.home() / "data" 
    done_tag = local_dir / "data_preparation_done"
    if done_tag.exists():
        print(f"Data preparation already done on {hostname}, skipping.")
        return
    remote_dir = get_region_storage()
    print(f"Remote directory: {remote_dir}")

    rel_dirs = [
        # "gsm8k",
        # "ckp/hf_models/Qwen2.5-0.5B-Instruct",
        "ckp/hf_models/phi-libri_ft_m1000_p8_new-QpHq/5000_hf_merged",
        "ckp/hf_models/phi4_mm_bias_merged",
        # "ckp/hf_models/phi4_mm_bias",
        "librispeech_biasing/words",
        "librispeech_biasing/ref",
        "LibriSpeech/test-clean",
        "LibriSpeech/train-clean-360/115/122944",
    ]

    for rel_dir in rel_dirs:
        print(f"Syncing directory: {rel_dir}")
        cmd = [
            "bbb", "sync", "--concurrency", "64",
            f"{remote_dir}/{rel_dir}", f"{local_dir}/{rel_dir}"
        ]
        subprocess.run(cmd, check=True)

    rel_files = [
        "LibriSpeech/ls_30k_shuf.tsv",
        "LibriSpeech/debug.tsv",
    ]
    for rel_file in rel_files:
        print(f"Syncing file: {rel_file}")
        cmd = [
            "bbb", "cp",
            f"{remote_dir}/{rel_file}", f"{local_dir}/{rel_file}"
        ]
        subprocess.run(cmd, check=True)
    print("Data preparation completed.")
    done_tag.touch()

def update_env_in_yaml(src_yaml_path, dst_yaml_path):
    """
    Reads a YAML file, substitutes environment variables in its content,
    and writes the result to a new YAML file.
    """
    with open(src_yaml_path, "r") as f:
        content = f.read()
    new_content = os.path.expandvars(content)
    with open(dst_yaml_path, "w") as f:
        f.write(new_content)

@ray.remote
def launch_training(config_file):
    """
    Launch training using the specified YAML config file.
    This function replicates the logic of the original shell script.
    """
    config_file = Path(config_file).expanduser().resolve()
    new_config_file = config_file.with_suffix(".tmp.yaml")
    update_env_in_yaml(config_file, new_config_file)
    
    os.chdir(Path(__file__).parent)
    print(f"Working Dir: {os.getcwd()}")
    

    output_dir = Path().home() / "outputs"
    os.makedirs(output_dir, exist_ok=True)

    rank = int(os.environ.get("RCALL_INSTANCE_INDEX", "0"))
    rank_size = int(os.environ.get("RCALL_INSTANCE_COUNT", "1"))
    num_gpu = int(os.environ.get("RCALL_NUM_GPU", "8"))
    job_name = os.environ.get("RCALL_JOB_NAME", None)
    assert job_name is not None, "RCALL_JOB_NAME must be set"
    main_process_ip = f"{job_name}-0" # head node IP
    main_process_port = 12345

    cmd = [
        "accelerate", "launch",
        "--num_processes", str(num_gpu * rank_size),
        "--num_machines", str(rank_size),
        "--machine_rank", str(rank),
        "--main_process_ip", str(main_process_ip),
        "--main_process_port", str(main_process_port),
        "trl/scripts/grpo_bias.py",
        "--config", str(new_config_file),
        "--output-dir", str(output_dir)
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


def main(config_file):
    """ Launch the job on all nodes by preparing the environment and data."""
    print("Preparing environment on all nodes...")
    run_nodes(prepare_environment)
    print("Preparing data on all nodes...")
    run_nodes(prepare_data)
    config_file = Path(config_file).absolute()
    print(f"Launch training with {config_file}...")
    run_nodes(launch_training, str(config_file))
    print("Job completed on all nodes.")
    
if __name__ == "__main__":
    ray.init(address="auto")  # Connect to the running cluster
    fire.Fire(main)
    # Example usage: python launch_job.py --config_file="path/to/config.yaml"