#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os
from pathlib import Path
import fire


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
            print(f"Node {node['NodeName']} is not alive, skipping.")
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
def job_log(cmd="tail", n=100, log_dir=None):
    log_dir = str(log_dir or os.environ.get("RCALL_LOGDIR", Path.home() / "results/*"))
    cmd = f"{cmd} -n {n}  {log_dir}/*.log"
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

    def sync_folder(self, folder):
        """Sync output directories across all Ray nodes."""
        folder = str(folder or Path.home() / "outputs")
        run_nodes(sync_folder, folder)

    def list_nodes(self):
        """List all nodes in the Ray cluster."""
        list_nodes()

    def log(self, cmd="tail", n=100, log_dir=None):
        """Tail logs from all Ray nodes."""
        run_nodes(job_log, cmd, n, log_dir)

    def run(self, *args, **kwargs):
        """Run a command on all Ray nodes."""
        cmd = " ".join(args)
        for k, v in kwargs.items():
            cmd += f" --{k} {v}"
        print(f"Running: {cmd}")
        run_nodes(ray.remote(run_cmd), cmd)


if __name__ == "__main__":
    """Main entry point for the RayTool."""
    fire.Fire(RayTool)
    # Example usage: python ray_tool.py run_nodes --fun=some_function --args=arg1,arg2
    # This will initialize Ray and run the specified function on all nodes.
