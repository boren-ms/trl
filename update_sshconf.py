#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import os
from sshconf import read_ssh_config_file
import subprocess
from io import StringIO
import pandas as pd
import fire
from pathlib import Path


def get_brix_instances():
    """Get the list of brix instances using the rcall-brix command."""
    cmd = ["rcall-brix", "ls", "--instances"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return pd.read_fwf(StringIO(result.stdout))
    return None


HOST_TEMP = {
    "HostName": "unknown",
    "IdentitiesOnly": "yes",
    "StrictHostKeyChecking": "no",
    "LogLevel": "ERROR",
    "UserKnownHostsFile": "/dev/null",
    "ControlMaster": "auto",
    "ControlPersist": "28800s",
    "IdentityFile": "/home/boren/.ssh/keys/orange/CLUSTER/boren/id_rsa",
    "ServerAliveInterval": "5",
    "ServerAliveCountMax": "3",
    "Port": "31338",
    "User": "root",
}


def update_ssh_config(df, ssh_conf, new_ssh_conf=None, wsl=False):
    cf = read_ssh_config_file(ssh_conf)
    user = os.getenv("USER", "boren")
    identity_fmt = "/home/{user}/.ssh/keys/orange/{cluster}/{user}/id_rsa"
    if wsl:
        identity_fmt = '"C:\\Users\\{user}\\.ssh\\keys\\orange\\{cluster}\\{user}\\id_rsa"'

    for _, row in df.iterrows():
        host = row["NAME"]
        cluster = row["CLUSTER"]
        host_name = f"{host}.rcall.{user}.svc.{cluster}.dev.openai.org"
        status = row["STATUS"]
        if pd.notna(row["IP"]) and row["STATUS"] != "Running":
            host_name = row["IP"]
        if host in cf.hosts():
            print(f"Update {host} [{status}]: {host_name}")
            cf.set(host, HostName=host_name)
        else:
            print(f"Add {host} [{status}]: {host_name}")
            host_kwargs = HOST_TEMP.copy()
            host_kwargs["HostName"] = host_name
            host_kwargs["IdentityFile"] = identity_fmt.format(user=user, cluster=cluster)
            cf.add(host, before_host=cf.hosts()[0], **host_kwargs)
    new_ssh_conf = new_ssh_conf or ssh_conf
    cf.write(new_ssh_conf)
    return new_ssh_conf


def main(ssh_conf=None, wsl=False):
    """Main function to update the SSH config."""
    user = os.getenv("USER", "boren")
    df = get_brix_instances()
    if df is not None:
        ssh_confs = [(ssh_conf, wsl)] if ssh_conf else [(f"/home/{user}/.ssh/config", False), (f"/mnt/c/Users/{user}/.ssh/config", True)]
        for ssh_conf, wsl in ssh_confs:
            print("=" * 20)
            print(f"Updating SSH: {ssh_conf}")
            if not Path(ssh_conf).exists():
                print(f"Skipping {ssh_conf} due to missing.")
                continue
            update_ssh_config(df, ssh_conf, wsl=wsl)
            print(f"Updated SSH: {ssh_conf}")
    else:
        print("Failed to retrieve brix instances.")


# %%
# main()
# %%
if __name__ == "__main__":
    fire.Fire(main)
