#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import ray
import fire
import subprocess


def restart_node(node):
    """Attempt to restart a Ray node if it's local."""
    addr = node['NodeManagerAddress']
    # Only attempt restart if the node is local
    if addr == "127.0.0.1" or addr == "localhost":
        print(f"   Attempting to restart local node at {addr}...")
        try:
            subprocess.run(["ray", "stop"], check=True)
            subprocess.run(["ray", "start", "--head"], check=True)
            print("   Node restart command issued.")
        except Exception as e:
            print(f"   Failed to restart node: {e}")
    else:
        print(f"   Node at {addr} is not local. Please restart it manually.")


def check_ray():
    """Check the connection to a Ray cluster and print the status of nodes."""
    print("Connecting to Ray cluster...")
    ray.init(address="auto")  # Connect to the running cluster
    nodes = ray.nodes()
    print(f"Found {len(nodes)} nodes in the cluster:")
    for node in nodes:
        print(f" - {node['NodeName']}[{node['NodeManagerAddress']}] (Alive: {node['Alive']})")
        # if not node['Alive']:
        #     print(f"   Resources: {node['Resources']}")
        #     restart_node(node)


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(check_ray)