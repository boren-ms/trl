#%%
import subprocess
import fire
from typing import Any

import rcall.brix
import rcall.cli
KUBE_CLUSTERS=['prod-uksouth-7', 'prod-uksouth-8', 'prod-uksouth-15', 'prod-southcentralus-hpe-2', 'prod-southcentralus-hpe-5', 'prod-southcentralus-hpe-3', 'prod-westus2-19', 'prod-southcentralus-hpe-4']
#%%
pods = rcall.brix.find_resources(
    "boren", KUBE_CLUSTERS, [], "pool"
)
for pod in pods:
    print(pod)

#%%
def command_pssh(args: Any, conf: Any, clusters: list[str]):
    """Run a command across multiple instances in parallel"""
    pods = rcall.brix.find_resources(
        conf.USER_NAME, clusters, args.pattern, "pod", condition="Ready"
    )
    if len(pods) == 0:
        print("No resources found.")
        return
    assert args.remote_cmd, "missing command, did you use `-- <cmd>`?"
    if args.direct:
        names = [pod["metadata"]["name"] for pod in pods]
        ssh_commands = [rcall.brix.ssh_command(pod) for pod in pods]
        rcall.cli.pssh_direct(names, ssh_commands, args.remote_cmd)
    else:
        instances = {}
        for pod in pods:
            instances[pod["metadata"]["name"]] = dict(
                ssh_command=rcall.brix.ssh_command(pod),
                cluster=pod["cluster"],
                pod_ip=pod["status"]["podIP"],
            )
        rcall.cli.pssh_tree(
            instances,
            args.remote_cmd,
            print_successes=args.print,
            initial_instances_per_cluster=args.initial_instances_per_cluster,
        )

# Define your jobs as a dictionary: {job_name: command}
jobs = {"list_home": "ls ~", "show_date": "date", "print_working_dir": "pwd"}


def run_jobs(jobs_dict):
    for name, cmd in jobs_dict.items():
        print(f"Running job: {name}")
        try:
            result = subprocess.run(
                cmd, shell=True, check=True, text=True, capture_output=True
            )
            print(f"Output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error running job '{name}': {e}\n{e.stderr}")


if __name__ == "__main__":
    fire.Fire(run_jobs)
