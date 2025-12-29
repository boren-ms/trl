import os


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


def get_storage(region=None):
    region = region or get_region()
    return REGION_STORAGES.get(region, "orngcresco")


def is_orng_path(path):
    orng_pfxs = [f"az://{s}/" for s in REGION_STORAGES.values()]
    return any(path.startswith(pfx) for pfx in orng_pfxs)


def to_orng(path, storage=None, log=False):
    storage = storage or get_storage()
    mappings = {
        "az://oaidatasets2/": f"az://{storage}/models/mm/oaidatasets2/",
    }
    for s in REGION_STORAGES.values():
        mappings[f"az://{s}/"] = f"az://{storage}/"
    org_path = path
    for k, v in mappings.items():
        path = path.replace(k, v)
    if log:
        print(f"OrangePath: {org_path} => {path}")
    return path
