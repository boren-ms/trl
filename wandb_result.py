#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import wandb
import fire
import os
import urllib.parse


def get_run(path):
    parsed = urllib.parse.urlparse(path)
    path = parsed.path.replace("/runs/", "/").strip("/")
    assert path is not None, "Either url or path must be provided to get_run"
    try:
        return wandb.Api().run(path)
    except wandb.Error:
        return None


def get_run_result(runs, prefix="metric"):
    results = []
    for run in runs:
        run_dict = {"name": run.name}
        for key, value in run.summary.items():
            if key and key.startswith(prefix):
                new_key = key.split("/", 1)[-1]  # Remove prefix
                run_dict[new_key] = value
        results.append(run_dict)
    df = pd.DataFrame(results)
    df.set_index("name", inplace=True)
    df = df.T
    if df.empty:
        return None
    df[["dataset", "#bias", "metric"]] = df.index.to_series().str.split("_", n=2, expand=True)
    df["#bias"] = pd.to_numeric(df["#bias"], errors="coerce").fillna(0).astype(int)  # Convert to int
    df = df[df["metric"].isin(["WER", "BWER", "UWER"])]
    df = df.sort_values(by=["dataset", "#bias", "metric"], ascending=[True, True, False])
    df = df.drop(columns=["dataset", "#bias", "metric"])
    return df


class RunChecker:
    def __init__(self, entity=None, project=None, metric="metric", save_excel=False, excel_path=None):
        self.host = os.environ.get("WANDB_ORGANIZATION", "https://msaip.wandb.io")
        self.entity = entity or os.environ.get("WANDB_ENTITY", "genai")
        self.project = project or os.environ.get("WANDB_PROJECT", "biasing")
        self.metric = metric
        self.excel_path = excel_path or ("results.xlsx" if save_excel else None)
        #     wandb.login(key=os.environ.get("WANDB_API_KEY"), host=self.host)

    def check(self, run_url, key=None, nrows=10):
        run = get_run(run_url)
        if run is None:
            print(f"Run not found: {run_url}")
            return None
        df = get_run_result([run], prefix=self.metric)
        if df is None:
            print(f"No results [{self.metric}] found for run: {run_url}")
            return None
        if self.excel_path:
            print(f"writing to {self.excel_path}")
            df.to_excel(self.excel_path, index=True)
        if key:
            df = df[df.index.str.contains(key)]
        df = df.head(nrows)
        import pprint as pp

        pp.pprint(df.to_dict(orient="index"))

    def search(self, run_name, key=None, nrows=10):
        """search runs"""
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}", filters={"display_name": {"$regex": run_name}})
        if not runs:
            print(f"No runs found matching '{run_name}'")
            return
        print(f"Found {len(runs)} runs matching '{run_name}'")
        df = get_run_result(runs, prefix=self.metric)
        if df is None:
            print(f"No results [{self.metric}] found for runs matching '{run_name}'")
            return None
        if self.excel_path:
            print(f"Results written to {self.excel_path}")
            df.to_excel(self.excel_path)
        if key:
            df = df[df.index.str.contains(key)]
        df = df.head(nrows)
        import pprint as pp

        pp.pprint(df)


if __name__ == "__main__":
    fire.Fire(RunChecker)
