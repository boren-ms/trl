#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
import importlib
from pathlib import Path
import fire
import time
import blobfile as bf
import importlib.metadata
from ray_tool import run_nodes, run_cmd, head_node_label, init_ray, release_gpus, sync_folder, list_nodes


def to_int(value, default=-1):
    """Convert a value to an integer, if possible."""
    try:
        return int(value)
    except ValueError:
        return default

def chkp_index(name):
    """Extract the checkpoint index from a checkpoint directory name."""
    if not name.startswith("checkpoint-"):
        return -1
    return to_int(name.split("-")[-1], -1)

@ray.remote
class OutputWatcher:
    def __init__(self, local_dir, remote_dir, interval=600):
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.interval = interval
        self._running = True

    def sync_latest_chkp(self):
        """sync checkpoint folder from remote to local."""
        print("Syncing latest checkpoint from local to remote ...")
        if not Path(self.local_dir).exists():
            print(f"Local directory [{self.local_dir}] does not exist, skipping sync.")
            return
        for file_path in Path(self.local_dir).iterdir():
            if not file_path.is_file():
                continue
            print(f"Syncing file {file_path.name} to {self.remote_dir}/{file_path.name}")
            bf.copy(file_path, f"{self.remote_dir}/{file_path.name}", overwrite=True)
        
        chkp_dirs = [d for d in Path(self.local_dir).iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        chkp_dirs = sorted(chkp_dirs, key=lambda d: chkp_index(d.name), reverse=True)
        ckhps = [chkp_index(d.name) for d in chkp_dirs]
        if not chkp_dirs:
            print(f"No checkpoint found in {self.local_dir}.")
            return
        print(f"Found {len(chkp_dirs)} checkpoints in {self.local_dir}.")
        print("Latest 20 checkpoints: ", ckhps[:20])
        local_chkp_dir = chkp_dirs[0]
        print(f"Latest checkpoint: {local_chkp_dir}")
        remote_chkp_dir = f"{self.remote_dir}/{local_chkp_dir.relative_to(self.local_dir)}"
        cmd = ["bbb", "sync", "--concurrency", "64", f"{local_chkp_dir}/", f"{remote_chkp_dir}/"]
        print(f"Syncing latest checkpoint from {local_chkp_dir} to {remote_chkp_dir}")
        run_cmd(cmd)
        print("Sync completed.")

    def start(self):
        print(f"Watcher started with interval {self.interval/60} minutes.")
        print(f"Local dir: {self.local_dir}")
        print(f"Remote dir: {self.remote_dir}")
        while self._running:
            print("Watcher tick!")
            self.sync_latest_chkp()
            time.sleep(self.interval)

    def stop(self):
        self._running = False
        print("Watcher stopped.")


def is_package_version(package_name, target_version):
    """Check if the specified package is installed with the target version."""
    try:
        version = importlib.metadata.version(package_name)
        return version == target_version
    except importlib.metadata.PackageNotFoundError:
        return False


def get_region():
    """Get the region of the Kubernetes cluster from the environment variable."""
    rcall_kube_cluster = os.environ.get("RCALL_KUBE_CLUSTER", "")
    cluster_region = rcall_kube_cluster.split("-")[1] if "-" in rcall_kube_cluster else None
    return cluster_region


REGION_STORAGES = {
    "southcentralus": "orngscuscresco",
    "westus2": "orngwus2cresco",
    "uksouth": "orngcresco",
}


class UserStorage:
    """Class to manage user storage paths based on the region of the Kubernetes cluster."""

    def __init__(self, region=None):
        """Initialize the UserStorage with the specified region."""
        self.region = region or get_region()
        assert self.region, "Region must be specified or set in RCALL_KUBE_CLUSTER environment variable"
        self.region_storage = REGION_STORAGES.get(self.region, "orngscuscresco")
        self.user = os.environ.get("OPENAI_USER", "boren")

    @property
    def home_path(self):
        """Get the storage path based on the region."""
        return f"az://{self.region_storage}/data/{self.user}"

    @property
    def data_path(self):
        """Get the user data storage path based on the region."""
        return f"{self.home_path}/data"

    @property
    def output_path(self):
        """Get the user output storage path based on the region."""
        return f"{self.home_path}/outputs"


ORNG_USER = UserStorage()


@ray.remote
def prepare_environment(forced=False):
    """Prepare the environment on each node by installing necessary packages."""
    hostname = os.uname().nodename
    print(f"Preparing environment on node: {hostname}")
    packages = [
        "torch==2.6.0",
        "ray==2.36.1",
        "transformers==4.51.3",
        "vllm==0.8.5.post1",
    ]
    if all(is_package_version(*package.split("==")) for package in packages) and not forced:
        print(f"Required packages already installed on {hostname}, skipping installation.")
        return
    run_cmd("pip uninstall -y torch torchvision torchaudio transformers flash-attn vllm trl")
    run_cmd("uv pip install --system torch==2.6.0 torchvision torchaudio transformers==4.51.3  trl peft tensorboardX blobfile soundfile more-itertools whisper_normalizer fire")
    run_cmd("pip install vllm==0.8.5.post1 && pip install ray==2.36.1")
    run_cmd("pip install torch==2.6.0 flash-attn ")
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
    remote_dir = ORNG_USER.data_path
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
def prepare_local_output(local_dir, remote_dir):
    """Prepare output on each node by syncing from the remote storage."""
    hostname = os.uname().nodename
    print(f"Sync remote output on node: {hostname}")
    print(f"Remote output directory: {remote_dir}")
    print(f"Local output directory: {local_dir}")
    if not bf.exists(remote_dir) or not bf.isdir(remote_dir):
        print(f"Remote directory [{remote_dir}] does not exist.")
        return
    
    # sync remote files to local directory 
    for file_path in bf.scandir(remote_dir):
        if not file_path.is_file:
            continue
        local_file_path = Path(local_dir) / file_path.name
        if local_file_path.exists():
            print(f"File {local_file_path} already exists, skipping.")
            continue
        print(f"Syncing file {file_path.name} to {local_file_path}")
        # local_file_path.parent.mkdir(parents=True, exist_ok=True)
        # bf.copy(file_path, local_file_path)
        cmd = ["bbb", "cp", f"{remote_dir}/{file_path.name}", f"{local_file_path}"]
        run_cmd(cmd)
        
    # sync remote checkpoints to local directory
    chkps = [(chkp_index(d.name), d.name) for d in bf.scandir(remote_dir) if d.is_dir and chkp_index(d.name) >= 0]
    chkps = sorted(chkps, key=lambda x: x[0], reverse=True)
    if not chkps:
        print(f"No checkpoints found in {remote_dir}.")
        return
    print(f"Found {len(chkps)} checkpoints in {remote_dir}.")
    print("Latest 20 checkpoints: ", [chkp[0] for chkp in chkps[:20]])
    latest_chkp = chkps[0][1]
    print(f"Syncing latest checkpoint ({latest_chkp}) to local directory...")
    cmd = ["bbb", "sync", "--concurrency", "64", f"{remote_dir}/{latest_chkp}/", f"{local_dir}/{latest_chkp}/"]
    run_cmd(cmd)
    print("Data preparation completed.")


def update_envs(yaml_path):
    """Reads a YAML file, substitutes environment variables in its content"""
    print(f"Updating variables in {yaml_path}")
    os.environ["DATA_STORAGE"] = ORNG_USER.region_storage
    content = Path(yaml_path).read_text()
    expanded_content = os.path.expandvars(content)
    Path(yaml_path).write_text(expanded_content)


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


def get_output_dirs(rel_path=None, job_name=None):
    """Get the remote output directory based on the job name."""
    job_name = job_name or os.environ.get("RCALL_JOB_NAME", None)
    remote_output_dir = f"{ORNG_USER.output_path}/{job_name}"
    local_output_dir = Path.home() / "outputs"
    if rel_path:
        remote_output_dir = f"{remote_output_dir}/{rel_path}"
        local_output_dir = local_output_dir / rel_path
    return str(local_output_dir), remote_output_dir


def run_chkp_watcher(local_dir=None, remote_dir=None, interval=600):
    """Start the output watcher to sync outputs periodically."""
    head_node = head_node_label()
    print(f"Watching  @ {head_node} every {interval/60} minutes")
    print(f"Local directory: {local_dir}")
    print(f"Remote directory: {remote_dir}")
    watcher = OutputWatcher.options(resources={head_node: 0.01}).remote(local_dir=local_dir, remote_dir=remote_dir, interval=interval)
    watcher.start.remote()
    return watcher


def main(config_file, forced=False):
    """Launch the job on all nodes by preparing the environment and data."""
    init_ray()
    list_nodes()
    
    config_file = Path(config_file).absolute()
    print(f"Using config file: {config_file}")
    output_dir, remote_output_dir = get_output_dirs(config_file.stem)
    
    results = []
    print("Preparing environment on all nodes...")
    results += run_nodes(prepare_environment, forced=forced, waiting=False)

    print("Preparing data on all nodes...")
    results += run_nodes(prepare_data, forced=forced, waiting=False)

    print("Releasing GPUs on all nodes...")
    results += run_nodes(release_gpus, waiting=False)

    print("Preparing output on all nodes...")
    results += run_nodes(prepare_local_output, local_dir=output_dir, remote_dir=remote_output_dir, waiting=False)

    # Ensure all tasks are completed before proceeding
    ray.get(results)

    print("Syncing outputs from head to other nodes...")
    run_nodes(sync_folder, str(output_dir))

    print("Starting output watcher on head node...")
    watcher = run_chkp_watcher(local_dir=output_dir, remote_dir=remote_output_dir, interval=600)

    print(f"Launch training with {config_file}...")
    run_nodes(launch_training, str(config_file), output_dir=str(output_dir))
    print("Job completed on all nodes.")
    ray.get(watcher.stop.remote())
    print("All tasks completed, stopping watcher.")


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(main)
    # Example usage: python launch_job.py --config_file="path/to/config.yaml"
