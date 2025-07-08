#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
import importlib
from pathlib import Path
import fire
import time
import importlib.metadata

def run_cmd(cmd, check=True):
    """Run a shell command and print it."""
    if isinstance(cmd, (list, tuple)):
        cmd = " ".join(cmd)
    print(f"Running: {cmd}")
    ret = subprocess.run(cmd, shell=True, check=check)
    print(f"Cmd: {cmd} returned: {ret.returncode}")
    return ret


def head_hostname():
    """Get the head node hostname from environment variables."""
    job_name = os.environ.get("RCALL_JOB_NAME", None)
    assert job_name is not None, "RCALL_JOB_NAME must be set"
    return f"{job_name}-0"  # head node IP


def run_nodes(fun, *args, waiting=True, **kwargs):
    nodes = ray.nodes()
    # node_names = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]
    node_names = [node["NodeName"] for node in nodes if node["Alive"]]

    # Launch one task per node, each pinned to a specific node
    results = []
    for node_name in node_names:
        # Use custom resource label to ensure the function runs on this node
        # Each node has a resource label 'node:<ip>'
        node_label = f"node:{node_name}"
        result = fun.options(resources={node_label: 0.01}).remote(*args, **kwargs)
        results.append(result)
    if waiting:
        results = ray.get(results)
    return results


@ray.remote
class OutputWatcher:
    def __init__(self, local_dir, remote_dir, interval=600):
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.interval = interval
        self._running = True

    def start(self):
        print(f"Watcher started with interval {self.interval} seconds.")
        print("Watching for output changes, and syncing if necessary.")
        print(f"Local directory: {self.local_dir}")
        print(f"Remote directory: {self.remote_dir}")
        while self._running:
            # Place your watched logic here
            print("Watcher tick!")
            cmd = [
                "bbb",
                "sync",
                "--concurrency",
                "64",
                f"{self.local_dir}/",
                f"{self.remote_dir}/",
            ]
            run_cmd(cmd)
            print("Sync completed.")
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        print("Watcher stopped.")


REGION_STORAGES = {
    "southcentralus": "orngscuscresco",
    "westus2": "orngwus2cresco",
    "uksouth": "orngukscresco",
}



def is_package_version(package_name, target_version):
    """Check if the specified package is installed with the target version."""
    try:
        version = importlib.metadata.version(package_name)
        return version == target_version
    except importlib.metadata.PackageNotFoundError:
        return False

    
def get_region_storage():
    """Get the storage path based on the region of the Kubernetes cluster."""
    rcall_kube_cluster = os.environ.get("RCALL_KUBE_CLUSTER", "")
    cluster_region = rcall_kube_cluster.split("-")[1] if "-" in rcall_kube_cluster else ""
    data_storage = REGION_STORAGES.get(cluster_region, "orngscuscresco")
    return data_storage


def get_remote_data_dir():
    """Get the storage path based on the region of the Kubernetes cluster."""
    data_storage = get_region_storage()
    user = os.environ.get("RCALL_USER", "boren")
    return f"az://{data_storage}/data/{user}/data"


@ray.remote
def prepare_environment(forced=False):
    """Prepare the environment on each node by installing necessary packages."""
    hostname = os.uname().nodename
    print(f"Preparing environment on node: {hostname}")
    packages = [
        "torch==2.6.0",
        "ray==2.36.1",
        "transformers==4.51.3",
    ]
    if all(is_package_version(*package.split("==")) for package in packages) and not forced:
        print(f"Required packages already installed on {hostname}, skipping installation.")
        return
    run_cmd("pip uninstall -y torch torchvision torchaudio transformers flash-attn vllm trl")
    run_cmd("uv pip install --system torch==2.6.0 ray==2.36.1 torchvision torchaudio transformers==4.51.3  trl peft tensorboardX blobfile soundfile more-itertools whisper_normalizer fire")
    run_cmd("pip install torch==2.6.0 flash-attn ")
    run_cmd("pip install torch==2.6.0 vllm==0.8.5.post1 --no-deps")
    run_cmd("pip uninstall -y trl")
    run_cmd("pip install -e /root/code/trl --no-deps")
    print("Environment preparation completed.")


@ray.remote
def prepare_data(forced=False):
    """Prepare data on each node by syncing from the remote storage."""
    hostname = os.uname().nodename
    print(f"Preparing data on node: {hostname}")
    local_dir = Path.home() / "data"
    done_tag = local_dir / "data_preparation_done"
    if done_tag.exists() and not forced:
        print(f"Data preparation already done on {hostname}, skipping.")
        return
    remote_dir = get_remote_data_dir()
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
        cmd = ["bbb", "sync", "--concurrency", "64", f"{remote_dir}/{rel_dir}", f"{local_dir}/{rel_dir}"]
        run_cmd(cmd)

    rel_files = [
        "LibriSpeech/ls_30k_shuf.tsv",
        "LibriSpeech/debug.tsv",
    ]
    for rel_file in rel_files:
        print(f"Syncing file: {rel_file}")
        cmd = ["bbb", "cp", f"{remote_dir}/{rel_file}", f"{local_dir}/{rel_file}"]
        run_cmd(cmd)
    print("Data preparation completed.")
    done_tag.touch()


@ray.remote
def sync_outputs(output_dir):
    """Sync the output files from the remote storage."""
    head_node = head_hostname()
    cur_node = os.uname().nodename
    if cur_node == head_node:
        print(f"Skipping checkpoint sync on head node: {cur_node}")
        return
    print(f"Syncing checkpoints from head node: {head_node} to current node: {cur_node}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avz", f"{head_node}:{output_dir}/", f"{output_dir}/"]
    run_cmd(cmd)
    print("Output syncing completed.")


def update_envs(yaml_path):
    """Reads a YAML file, substitutes environment variables in its content"""
    print(f"Updating variables in {yaml_path}")
    os.environ["DATA_STORAGE"] = get_region_storage()
    content = Path(yaml_path).read_text()
    expanded_content = os.path.expandvars(content)
    Path(yaml_path).write_text(expanded_content)


@ray.remote
def launch_training(config_file):
    """Launch training using the specified YAML config file."""
    config_file = Path(config_file).absolute()
    update_envs(config_file)

    os.chdir(Path(__file__).parent)
    print(f"Working Dir: {os.getcwd()}")
    output_dir = Path().home() / "outputs"
    os.makedirs(output_dir, exist_ok=True)

    rank = int(os.environ.get("RCALL_INSTANCE_INDEX", "0"))
    rank_size = int(os.environ.get("RCALL_INSTANCE_COUNT", "1"))
    num_gpu = int(os.environ.get("RCALL_NUM_GPU", "8"))
    job_name = os.environ.get("RCALL_JOB_NAME", None)
    assert job_name is not None, "RCALL_JOB_NAME must be set"
    main_process_ip = f"{job_name}-0"  # head node IP
    main_process_port = 12345

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
        "trl/scripts/grpo_bias.py",
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


def run_output_watcher():
    """Start the output watcher to sync outputs periodically."""
    output_dir = Path.home() / "outputs"
    remote_output_dir = f"{get_remote_data_dir()}/outputs/{os.environ.get('RCALL_JOB_NAME', 'UnknownJob')}"
    watcher = OutputWatcher.options(resources={f"hostname:{head_hostname()}": 0.01}).remote(
        local_dir=output_dir,
        remote_dir=remote_output_dir,
        interval=1800,  # changed from 600 to 1800 (30min)
    )
    print("Starting output watcher...")
    watcher.start.remote()

@ray.remote
def release_gpus():
    """Release GPUs on the current node."""
    hostname = os.uname().nodename
    print(f"Releasing GPUs on node: {hostname}")
    list_cmd = "lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u"
    kill_cmd = "lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u | xargs -I {} kill -9 {}"
    print("Listing processes using NVIDIA devices:")
    run_cmd(list_cmd, check=False)
    print("Killing processes using NVIDIA devices:")
    run_cmd(kill_cmd, check=False)
    print("List processes using NVIDIA devices again:")
    run_cmd(list_cmd, check=False)
    print("GPUs released.")

def main(config_file, forced=False):
    """Launch the job on all nodes by preparing the environment and data."""
    results = []
    
    print("Preparing environment on all nodes...")
    results += run_nodes(prepare_environment, forced=forced, waiting=False)
    
    print("Preparing data on all nodes...")
    results += run_nodes(prepare_data, forced=forced, waiting=False)

    print("Syncing outputs on all nodes...")
    results += run_nodes(sync_outputs, str(Path.home() / "outputs"), waiting=False)
    
    print("Releasing GPUs on all nodes...")
    results += run_nodes(release_gpus, waiting=False)
    
    # Ensure all tasks are completed before proceeding
    ray.get(results)
    print("Starting output watcher on head node...")
    run_output_watcher()
    
    config_file = Path(config_file).absolute()
    print(f"Launch training with {config_file}...")
    run_nodes(launch_training, str(config_file))
    print("Job completed on all nodes.")


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    print("Connecting to Ray cluster...")
    ray.init(address="auto")  # Connect to the running cluster
    nodes = ray.nodes()
    print(f"Found {len(nodes)} nodes in the cluster:")
    for node in nodes:
        print(f" - {node['NodeName']}[{node['NodeManagerAddress']}] (Alive: {node['Alive']})")
    fire.Fire(main)
    # Example usage: python launch_job.py --config_file="path/to/config.yaml"
