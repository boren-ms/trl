# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wandb utilities for TRL scripts."""

import json
import os
from pathlib import Path

import wandb

from .job_utils import get_job_name


def save_run_info(run, work_dir=None, file_name="run_info.json"):
    """Save run identifying information to a JSON file."""
    if work_dir is None:
        return
    info_file = Path(work_dir) / file_name
    info_file.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
    }
    json.dump(info, info_file.open("w"), indent=2)
    print(f"Run info saved to {info_file}")


def load_run_info(work_dir=None, file_name="run_info.json"):
    """Load run info from JSON and resume the run."""
    if work_dir is None:
        return {}
    info_file = Path(work_dir) / file_name
    if not info_file.exists():
        print(f"Run info file {file_name} does not exist in {work_dir}.")
        return {}
    print(f"Loading run info from {info_file}")
    info = json.load(info_file.open("r"))
    url = info.get("run_url", "")
    print(f"Reuse run: {url}")

    parts = Path(url).parts
    if url.startswith("https://msaip.wandb.io/") and len(parts) > 4:
        print("Run info from", url)
        info["run_id"] = info.get("run_id", parts[-1])
        info["project"] = info.get("project", parts[-3])
        info["entity"] = info.get("entity", parts[-4])
    return info


def init_wandb(job_name=None, project=None, config=None, output_dir=None, skip_run_info=False):
    """Initialize wandb."""
    project = os.environ.get("WANDB_PROJECT", project or "biasing")
    job_name = get_job_name(job_name)
    print(f"Project Name: {project}, Run Name: {job_name}")
    key = os.environ.get("WANDB_API_KEY", "")
    host = os.environ.get("WANDB_ORGANIZATION", "")
    wandb.login(host=host, key=key, relogin=True)
    entity = os.environ.get("WANDB_ENTITY", "genai")
    run_info = {} if skip_run_info else load_run_info(output_dir)
    run = wandb.init(
        entity=run_info.get("entity", entity),
        project=run_info.get("project", project),
        id=run_info.get("run_id", None),
        name=run_info.get("run_name", job_name),
        resume="allow",
        config=run_info.get("config", config),
    )
    print("wandb offline: ", run.settings._offline)  # Should be True
    print("wandb mode: ", run.settings.mode)  # Should be "offline"
    save_run_info(run, output_dir)