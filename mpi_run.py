#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import sys

def launch_mpirun(nodes, cmd):
    n_nodes = len(nodes)
    hostlist = ",".join(nodes)
    mpi_cmd = ["mpirun", "-l", "-np", str(n_nodes), "--host", hostlist, *cmd]
    print("Running:", " ".join(mpi_cmd))
    subprocess.run(mpi_cmd)

def find_nodes(skip_head=False):
    n = int(os.environ["RCALL_INSTANCE_COUNT"])
    job_name = os.environ["RCALL_JOB_NAME"]
    s = int(skip_head)
    return [f"{job_name}-{i}" for i in range(s, n)]

def main():
    args = sys.argv[1:]
    skip_head = "--skip_head" in args
    if skip_head:
        args.remove("--skip_head")
    if not args:
        print("Usage: python mpi_run.py <command> [args...]")
        sys.exit(1)
    nodes = find_nodes(skip_head)
    if not nodes:
        print("No nodes found. Please check your environment variables.")
        sys.exit(1)
    print(f"Found [{len(nodes)}] nodes:", nodes)
    print("Command to run:", args)
    launch_mpirun(nodes, args)

if __name__ == "__main__":
    main()
  