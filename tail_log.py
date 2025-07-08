#!/usr/bin/env python
from ray_utils import run_nodes, run_cmd, init_ray
import os
import ray
import fire

@ray.remote
def tail_logs():
    log_dir = os.environ.get("RCALL_LOGDIR", f"/root/results/{ray.get_runtime_context().job_id}")
    cmd = f'tail -n 100 -f {log_dir}/*.log'
    print(f"Tailing logs in {log_dir} with command: {cmd}")
    run_cmd(cmd, check=True)
    
def main():
    """Main entry point for tailing logs across all Ray nodes."""
    print("Starting to tail logs on all Ray nodes...")
    init_ray()
    run_nodes(tail_logs, waiting=False)

if __name__ == "__main__":
    fire.Fire(main)