#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import ray
import os

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



def init_ray():
    """Check the connection to a Ray cluster and print the status of nodes."""
    print("Connecting to Ray cluster...")
    ray.init(address="auto")  # Connect to the running cluster
    nodes = ray.nodes()
    print(f"Found {len(nodes)} nodes in the cluster:")
    for node in nodes:
        print(f" - {node['NodeName']}[{node['NodeManagerAddress']}] (Alive: {node['Alive']})")
    return nodes