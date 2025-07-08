#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import sys

def launch_mpirun(nodes, cmd):
    """
    Launch mpirun with N nodes, 1 process per node.
    Args:
        nodes (list): List of node hostnames or IPs.
    """
    n_nodes = len(nodes)
    hostlist = ",".join(nodes)
    mpi_cmd = [
        "mpirun",
        "-l",
        "-np", str(n_nodes),
        "--host", hostlist,
        *cmd,
    ]
    print("Running:", " ".join(mpi_cmd))
    subprocess.run(mpi_cmd)

def find_nodes():
    """
    Find the nodes in the current environment.
    This function assumes that the node names are in the format '{job_name}-{i}'.
    """
    n = int(os.environ["RCALL_INSTANCE_COUNT"])
    job_name = os.environ["RCALL_JOB_NAME"]
    return [f"{job_name}-{i}" for i in range(n)]

# Example usage:
if __name__ == "__main__":
    nodes = find_nodes()
    print(f"Found [{len(nodes)}] nodes:", nodes)
    cmd = sys.argv[1:]
    if not cmd:
        print("Usage: python mpi_run.py <command> [args...]")
        exit(1)
    print("Command to run:", cmd)
    launch_mpirun(nodes, cmd)

