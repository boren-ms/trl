# %%
import pandas as pd
import wandb
import fire
import os
import urllib.parse

def get_run(path):
    if path.startswith("http://"):
        parsed = urllib.parse.urlparse(path)
        path = parsed.path.replace("/runs/", "/").strip("/")
    assert path is not None, "Either url or path must be provided to get_run"
    return wandb.Api().run(path)

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
    df[["dataset", "#bias", "metric"]] = df.index.to_series().str.split("_", n=2, expand=True)
    df["#bias"] = pd.to_numeric(df["#bias"], errors="coerce").fillna(0).astype(int) # Convert to int
    df = df[df["metric"].isin(["WER", "BWER", "UWER"])]
    df = df.sort_values(by=["dataset", "#bias", "metric"], ascending=[True, True, False])
    # Remove columns "dataset", "#bias", "metric"
    df = df.drop(columns=["dataset", "#bias", "metric"])
    return df

class RunChecker:
    def __init__(self, entity=None, project=None):
        self.host = os.environ.get("WANDB_ORGANIZATION", "https://msaip.wandb.io")
        self.entity = entity or os.environ.get("WANDB_ENTITY", "genai")
        self.project = project or os.environ.get("WANDB_PROJECT", "biasing")
        wandb.login(key=os.environ.get("WANDB_API_KEY"), host=self.host, relogin=True)

    def check_run(self, run_url, excel_path=None):
        run = get_run(run_url)
        if run is None:
            print(f"Run not found: {run_url}")
            return None
        df = get_run_result([run])
        print(df)
        if excel_path:
            print(f"writing to {excel_path}")
            df.to_excel(excel_path, index=True)
    def search_runs(self, run_name, excel_path=None):
        """search runs"""
        api = wandb.Api()
        runs = api.runs(f"{self.entity}/{self.project}", filters={"name": run_name})
        df = get_run_result(runs)
        print(df)
        if excel_path:
            df.to_excel(excel_path, index=False)
            print(f"Results written to {excel_path}")
        return df


if __name__ == "__main__":
    fire.Fire(RunChecker)
