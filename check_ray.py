#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import ray
import fire


def check_ray():
    """Check the connection to a Ray cluster and print the status of nodes."""
    print("Connecting to Ray cluster...")
    ray.init(address="auto")  # Connect to the running cluster
    nodes = ray.nodes()
    print(f"Found {len(nodes)} nodes in the cluster:")
    for node in nodes:
        print(f" - {node['NodeName']}[{node['NodeManagerAddress']}] (Alive: {node['Alive']})")


if __name__ == "__main__":
    """Main entry point for launching the job on a Ray cluster."""
    fire.Fire(check_ray)