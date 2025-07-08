#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import fire

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

def find_nodes(skip_head=False):
    """
    Find the nodes in the current environment.
    This function assumes that the node names are in the format '{job_name}-{i}'.
    """
    n = int(os.environ["RCALL_INSTANCE_COUNT"])
    job_name = os.environ["RCALL_JOB_NAME"]
    s = int(skip_head)
    return [f"{job_name}-{i}" for i in range(s,n)]

def main(*cmd, skip_head=False):
    """
    Main function to run the script.
    It finds the nodes and launches mpirun with the provided command.
    """
    nodes = find_nodes(skip_head=skip_head)
    if not nodes:
        print("No nodes found. Please check your environment variables.")
        exit(1)
    
    if not cmd:
        print("Usage: python mpi_run.py <command> [args...]")
        exit(1)
    
    print(f"Found [{len(nodes)}] nodes:", nodes)
    print("Command to run:", cmd)
    launch_mpirun(nodes, cmd)


# Example usage:
if __name__ == "__main__":
    fire.Fire(main)


