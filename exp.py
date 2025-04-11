#!/usr/bin/env python
# %%
"""submit a job to AMLT"""
import os
from pathlib import Path
import fire
from amlt.api.project import ActiveProjectConfig
from amlt.api.registry import ProjectRegistry
from amlt.schema import JobSchema
import yaml


def get_project():
    """Get project"""
    project_config = ActiveProjectConfig()
    project = ProjectRegistry.get_project(project_config)
    return project


def get_job(exp_name, project=None):
    """Get job"""
    project = project or get_project()
    try:
        client = project.experiments.get(name=exp_name)
    except Exception as _:
        print(f"Experiment [{exp_name}] is not found")
        return None
    jobs = client.get_jobs()
    if len(jobs) == 0:
        print(f"Experiment [{exp_name}] has no jobs")
        return None
    job = jobs[0]
    return job


class Exp:
    """Experiment class"""

    def __init__(self):
        project_config = ActiveProjectConfig()
        self.project = ProjectRegistry.get_project(project_config)

    def _get_job(self, exp_name):
        """Get job"""
        return get_job(exp_name, self.project)

    def _job_output(self, exp_name):
        """Get job output"""
        job = self._get_job(exp_name)
        root_dir = Path("/datablob1")
        if job is None:
            user = os.getenv("USER", "phimm")
            return root_dir / f"projects/{user}/amlt-results/{exp_name}/"
        return root_dir / job.results_dir

    def show(self, exp_name):
        """Show job info"""
        job = self._get_job(exp_name)
        job_dict = JobSchema().dump(job)
        print(yaml.dump(job_dict, default_flow_style=False, sort_keys=False))

    def output(self, exp_name):
        """Get job output"""
        mnt_dir = self._job_output(exp_name)
        print(f"Exp: {exp_name}")
        print(f"Output: {mnt_dir}")
        chkps = [f.name for f in mnt_dir.iterdir() if f.is_dir()]
        print("Checkpoints: ", sorted(chkps, reverse=True))

    def latest(self, exp_name):
        """Get latest checkpoint"""
        mnt_dir = self._job_output(exp_name)
        chkp_list = [int(f.name) for f in mnt_dir.iterdir() if f.is_dir() and f.name.isdigit()]
        if len(chkp_list) == 0:
            print(f"Exp: {exp_name}")
            print(f"Output: {mnt_dir}")
            print(f"Checkpoints: {[f.name for f in mnt_dir.iterdir() if f.is_dir()]}")
            return
        latest_chkp = max(chkp_list)
        print(f"Exp: {exp_name}")
        print(f"Latest: {latest_chkp}")
        print(f"Output: {mnt_dir / str(latest_chkp)}")


if __name__ == "__main__":
    fire.Fire(Exp)
