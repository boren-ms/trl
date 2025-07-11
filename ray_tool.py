#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
from pathlib import Path
import fire
import time
import blobfile as bf
import importlib.metadata
from trl.data_utils import chkp_index


def upload_file(local_path, remote_path, overwrite=False):
    """Upload a file from local to remote storage."""
    local_mtime = Path(local_path).stat().st_mtime
    remote_mtime = None
    if bf.exists(remote_path) and not overwrite:
        print(f"Remote file {remote_path} already exists, skipping upload.")
        return
    try:
        if bf.exists(remote_path):
            remote_mtime = bf.stat(remote_path).mtime
    except Exception as e:
        print(f"Could not stat remote file {remote_path}: {e}")
    # Copy if remote does not exist or local is newer
    if remote_mtime is None or local_mtime > remote_mtime:
        print(f"Syncing file {local_path.name} to {remote_path} (local newer or remote missing)")
        bf.copy(local_path, remote_path, overwrite=True)
    else:
        print(f"Skipping {local_path.name}: remote is newer or same.")


@ray.remote
class OutputWatcher:
    def __init__(self, local_dir, remote_dir, interval=600, sync_all=True):
        self.local_dir = local_dir
        self.remote_dir = remote_dir
        self.interval = interval
        self._running = True
        self.sync_all = sync_all

    def sync_output_dir(self):
        """sync checkpoint folder from remote to local."""
        print("Syncing latest checkpoint from local to remote ...")
        if not Path(self.local_dir).exists():
            print(f"Local directory [{self.local_dir}] does not exist, skipping sync.")
            return
        if self.sync_all:
            print(f"Syncing all files from {self.local_dir} to {self.remote_dir}")
            cmd = ["bbb", "sync", "--concurrency", "64", f"{self.local_dir}/", f"{self.remote_dir}/"]
            run_cmd(cmd)
            return
        print(f"Syncing files expecting checkpoints from {self.local_dir} to {self.remote_dir}")
        cmd = ["bbb", "sync", "--concurrency", "64", f"{self.local_dir}/", f"{self.remote_dir}/", "--exclude", "checkpoint-*"]
        run_cmd(cmd)
        print("Syncing latest checkpoint ...")
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
            self.sync_output_dir()
            time.sleep(self.interval)
    
    def flush(self):
        """Flush the output directory by syncing it."""
        print("Flushing output directory...")
        self.sync_output_dir()
        print("Flush completed.")

    def stop(self, flush=True):
        """Stop the output watcher."""
        if flush:
            self.flush()
        self._running = False
        print("Watcher stopped.")


def run_output_watcher(local_dir=None, remote_dir=None, interval=600, sync_all=False):
    """Start the output watcher to sync outputs periodically."""
    head_node = head_node_label()
    print(f"Watching  @ {head_node} every {interval/60} minutes")
    print(f"Local directory: {local_dir}")
    print(f"Remote directory: {remote_dir}")
    watcher = OutputWatcher.options(resources={head_node: 0.01}).remote(local_dir=local_dir, remote_dir=remote_dir, interval=interval, sync_all=sync_all)
    watcher.start.remote()
    return watcher


def is_package_version(package_name, target_version):
    """Check if the specified package is installed with the target version."""
    try:
        version = importlib.metadata.version(package_name)
        return version == target_version
    except importlib.metadata.PackageNotFoundError:
        return False


def is_valid_model_path(model_dir):
    """Check if the model path is valid."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"Model path {model_dir} does not exist.")
        return False
    if not model_dir.is_dir():
        print(f"Model path {model_dir} is not a directory.")
        return False
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print(f"Config file {config_file} does not exist in the model directory.")
        return False
    if not any(model_dir.glob("*.safetensors")):
        print(f"No .safetensors files found in {model_dir}.")
        return False
    return True


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


def get_output_dirs(rel_path=None, job_name=None):
    """Get the remote output directory based on the job name."""
    job_name = job_name or os.environ.get("RCALL_JOB_NAME", None)
    remote_output_dir = f"{ORNG_USER.output_path}/{job_name}"
    local_output_dir = Path.home() / "outputs"
    if rel_path:
        remote_output_dir = f"{remote_output_dir}/{rel_path}"
        local_output_dir = local_output_dir / rel_path
    return str(local_output_dir), remote_output_dir


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


@ray.remote
def prepare_env(forced=False):
    """Prepare the environment on each node by installing necessary packages."""
    hostname = os.uname().nodename
    print(f"Preparing environment on node: {hostname}")
    packages = [
        "torch==2.6.0",
        "ray==2.36.1",
        "transformers==4.51.3",
        "vllm==0.8.5.post1",
        "trl==0.20.0.dev0",
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
        "ckp/hf_models/phi4_mm_bias",
        "ckp/hf_models/Phi-4-multimodal-instruct",
        "librispeech_biasing/words",
        "librispeech_biasing/ref",
        "LibriSpeech/test-clean",
        "LibriSpeech/test-other",
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


def update_envs(yaml_path):
    """Reads a YAML file, substitutes environment variables in its content"""
    print(f"Updating variables in {yaml_path}")
    os.environ["DATA_STORAGE"] = ORNG_USER.region_storage
    content = Path(yaml_path).read_text()
    expanded_content = os.path.expandvars(content)
    Path(yaml_path).write_text(expanded_content)


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


def head_node_label():
    """Get the head node IP address from environment variables."""
    nodes = ray.nodes()
    node_ip = nodes[0]["NodeManagerAddress"]
    return f"node:{node_ip}"


def run_nodes(fun, *args, waiting=True, **kwargs):
    nodes = ray.nodes()
    # Launch one task per node, each pinned to a specific node
    results = []
    for node in nodes:
        if not node["Alive"]:
            # print(f"Node {node['NodeName']} is not alive, skipping.")
            continue
        node_ip = node["NodeManagerAddress"]
        # Use custom resource label to ensure the function runs on this node
        # Each node has a resource label 'node:<ip>'
        node_label = f"node:{node_ip}"
        result = fun.options(resources={node_label: 0.01}).remote(*args, **kwargs)
        results.append(result)
    if waiting:
        results = ray.get(results)
    return results


def list_nodes():
    """List all nodes in the Ray cluster."""
    nodes = ray.nodes()
    print(f"Found {len(nodes)} nodes in the cluster:")
    for node in nodes:
        print(f" - {node['NodeName']}[{node['NodeManagerAddress']}] (Alive: {node['Alive']})")
    return nodes


def init_ray():
    """Check the connection to a Ray cluster and print the status of nodes."""
    print("Connecting to Ray cluster...")
    ray.init(address="auto")  # Connect to the running cluster
    print("Connected to Ray cluster.")


@ray.remote
def sync_folder(folder):
    """Sync the Folder from the remote storage."""
    head_node = head_hostname()
    cur_node = os.uname().nodename
    # Ensure the Folder exists for each node
    Path(folder).mkdir(parents=True, exist_ok=True)

    if cur_node == head_node:
        print(f"Skipping checkpoint sync on head node: {cur_node}")
        return
    print(f"Syncing checkpoints from head node: {head_node} to current node: {cur_node}")
    cmd = ["rsync", "-avz", f"{head_node}:{folder}/", f"{folder}/"]
    run_cmd(cmd)
    print("Folder syncing completed.")


@ray.remote
def release_gpus():
    """Release GPUs on the current node."""
    hostname = os.uname().nodename
    print(f"Releasing GPUs on node: {hostname}")
    list_cmd = "lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u"
    kill_cmd = "lsof /dev/nvidia* | awk '{print $2}' | grep -E '^[0-9]+$' | sort -u | xargs -I {} kill -9 {}"
    print("Listing processes using NVIDIA devices:")
    run_cmd(list_cmd)
    print("Killing processes using NVIDIA devices:")
    run_cmd(kill_cmd)
    print("List processes using NVIDIA devices again:")
    run_cmd(list_cmd)
    print("GPUs released.")


@ray.remote
def list_gpus():
    """List available GPUs on the current node."""
    cmd = "nvidia-smi | grep Default"
    print("Listing available GPUs:")
    run_cmd(cmd)
    print("GPUs listed.")


@ray.remote
def job_log(cmd="tail", key=None, n=100, log_dir=None):
    log_dir = str(log_dir or os.environ.get("RCALL_LOGDIR", Path.home() / "results/*"))
    pattern = f"*{key}*" if key else "*"
    cmd = f"{cmd} -n {n}  {log_dir}/{pattern}.log"
    print(f"Tailing logs in {log_dir} with command: {cmd}")
    run_cmd(cmd)


class RayTool:
    """A command-line tool for managing Ray clusters and nodes."""

    def __init__(self):
        """Initialize the RayTool class."""
        init_ray()
        print("Ray cluster initialized.")

    def list_gpus(self):
        """List available GPUs on the current node."""
        run_nodes(list_gpus)

    def release_gpus(self):
        """Release GPUs on all Ray nodes."""
        run_nodes(release_gpus)

    def sync_folder(self, folder=None):
        """Sync output directories across all Ray nodes."""
        folder = str(folder or Path.home() / "outputs")
        run_nodes(sync_folder, folder)

    def list_nodes(self):
        """List all nodes in the Ray cluster."""
        list_nodes()

    def log(self, cmd="tail", key=None, n=100, log_dir=None):
        """Tail logs from all Ray nodes."""
        run_nodes(job_log, cmd, key, n, log_dir)

    def run(self, *args, **kwargs):
        """Run a command on all Ray nodes."""
        cmd = " ".join(args)
        for k, v in kwargs.items():
            cmd += f" --{k} {v}"
        print(f"Running: {cmd}")
        run_nodes(ray.remote(run_cmd), cmd)

    def prepare_env(self, forced=False):
        """Prepare the environment on all Ray nodes by installing necessary packages."""
        print("Preparing environment on all nodes...")
        run_nodes(prepare_env, forced=forced)

    def prepare_data(self, forced=False):
        """Prepare data on all Ray nodes by syncing from the remote storage."""
        print("Preparing data on all nodes...")
        run_nodes(prepare_data, forced=forced)

    def prepare_local_output(self, rel_path=None):
        """Prepare output on all Ray nodes by syncing from the remote storage."""
        local_dir, remote_dir = get_output_dirs(rel_path)
        print(f"Preparing local output on all nodes: {local_dir} from {remote_dir}")
        run_nodes(prepare_local_output, local_dir, remote_dir)
        run_nodes(sync_folder, local_dir)

    def prepare_all(self, rel_path=None, forced=False):
        """Prepare the environment, data, and output on all Ray nodes."""
        results = []
        results += run_nodes(prepare_env, forced=forced, waiting=False)
        results += run_nodes(prepare_data, forced=forced, waiting=False)
        local_dir, remote_dir = get_output_dirs(rel_path)
        print(f"Preparing local output on all nodes: {local_dir} from {remote_dir}")
        results += run_nodes(prepare_local_output, local_dir, remote_dir, waiting=False)
        results = ray.get(results)
        self.sync_folder(local_dir)

    def run_output_watcher(self, rel_path=None, interval=600):
        """Run the output watcher on head."""
        local_dir, remote_dir = get_output_dirs(rel_path=rel_path)
        print(f"Running output watcher on head: {local_dir} from {remote_dir} every {interval/60} minutes")
        return run_output_watcher(local_dir, remote_dir, interval)


if __name__ == "__main__":
    """Main entry point for the RayTool."""
    fire.Fire(RayTool)
    # Example usage: python ray_tool.py run_nodes --fun=some_function --args=arg1,arg2
    # This will initialize Ray and run the specified function on all nodes.
