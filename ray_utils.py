
import subprocess
import ray

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    

def run_nodes(fun):
    ray.init(address="auto")  # Connect to the running cluster
    nodes = ray.nodes()
    node_ips = [node["NodeManagerAddress"] for node in nodes if node["Alive"]]

    # Launch one task per node, each pinned to a specific node
    results = []
    for node_ip in node_ips:
        # Use custom resource label to ensure the function runs on this node
        # Each node has a resource label 'node:<ip>'
        node_label = f"node:{node_ip}"
        result = fun.options(resources={node_label: 0.01}).remote(node_ip)
        results.append(result)
    return ray.get(results)