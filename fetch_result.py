# %%
import wandb
import fire
import os
import re
import urllib.parse

SELECTED_COLUMNS = ["name", "id", "config.learning_rate", "config.batch_size", "summary.best_accuracy", "summary.final_loss"]

# %%
host = os.environ.get("WANDB_ORGANIZATION", None)
key = os.environ.get("WANDB_API_KEY")
wandb.login(key=key, host=host, relogin=True)
# %%
import urllib.parse

url = "https://msaip.wandb.io/genai/biasing/runs/67fky69t"
parsed = urllib.parse.urlparse(url)
path_parts = parsed.path.strip("/").split("/")
# Expect path: <entity>/<project>/runs/<run_id>

entity = path_parts[0]
project = path_parts[1]
run_id = path_parts[3]
api = wandb.Api()
run = api.run(f"{entity}/{project}/{run_id}")

# %%
# refere to https://www.mongodb.com/docs/manual/reference/operator/query/
runs = api.runs(f"{entity}/{project}", filters={"display_name": {"$regex": "grpo_vllm"}})
#%%
for run in runs:
    print(f"Name: {run.name}, ID: {run.id}")


# %%
def get_nested(run, key):
    """Helper to get nested keys from run dict."""
    parts = key.split(".")
    val = run
    for part in parts:
        val = val.get(part, None) if isinstance(val, dict) else getattr(val, part, None)
        if val is None:
            break
    return val


def fetch_runs(entity, project, run_name=None):
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"name": run_name})
    results = []
    for run in runs:
        if run_name is not None and run.name != run_name:
            continue
        run_dict = {"name": run.name, "id": run.id, "config": dict(run.config), "summary": dict(run.summary)}
        row = [get_nested(run_dict, col) for col in SELECTED_COLUMNS]
        results.append(row)
    return results


def write_excel(rows, columns, filename):
    wb = Workbook()
    ws = wb.active
    ws.append(columns)
    for row in rows:
        ws.append(row)
    wb.save(filename)


def search_run_by_name(entity, project, run_name):
    """
    Search for a run by its name and return its details.
    Returns None if not found.
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"name": run_name})
    for run in runs:
        if run.name == run_name:
            return {"name": run.name, "id": run.id, "config": dict(run.config), "summary": dict(run.summary)}
    return None


def wandb_login_with_host(host):
    """
    Login to wandb with a specified host.
    """
    wandb.login(host=host)


def main(entity=None, project=None, run_name=None, output_dir=None):
    entity = entity or os.environ.get("WANDB_ENTITY", "your-entity")
    project = project or os.environ.get("WANDB_PROJECT", "your-project")
    rows = fetch_runs(entity, project, run_name)
    filename = "wandb_results.xlsx"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, filename)
    write_excel(rows, SELECTED_COLUMNS, filename)
    print(f"Results written to {filename}")


def search_run(entity=None, project=None, run_name=None):
    """
    Command-line function to search for a run by name and print its details.
    """
    entity = entity or os.environ.get("WANDB_ENTITY", "your-entity")
    project = project or os.environ.get("WANDB_PROJECT", "your-project")
    result = search_run_by_name(entity, project, run_name)
    if result:
        import pprint

        pprint.pprint(result)
    else:
        print(f"No run found with name: {run_name}")


def get_run_by_url(url):
    """
    Given a wandb run URL, fetch the run details.
    Example URL: https://msaip.wandb.io/genai/biasing/runs/67fky69t
    Returns: dict with run details or None if not found.
    """
    parsed = urllib.parse.urlparse(url)
    path_parts = parsed.path.strip("/").split("/")
    # Expect path: <entity>/<project>/runs/<run_id>
    if len(path_parts) < 4 or path_parts[-2] != "runs":
        print("Invalid wandb run URL format.")
        return None
    entity = path_parts[0]
    project = path_parts[1]
    run_id = path_parts[3]
    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        return {"name": run.name, "id": run.id, "config": dict(run.config), "summary": dict(run.summary)}
    except Exception as e:
        print(f"Error fetching run: {e}")
        return None


def fetch_run_by_url(url):
    """
    Command-line function to fetch and print run details by URL.
    """
    result = get_run_by_url(url)
    if result:
        import pprint

        pprint.pprint(result)
    else:
        print(f"No run found for URL: {url}")


def list_runs(entity=None, project=None):
    """
    List all wandb runs under a project.
    Prints run name and run id.
    """
    entity = entity or os.environ.get("WANDB_ENTITY", "your-entity")
    project = project or os.environ.get("WANDB_PROJECT", "your-project")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    for run in runs:
        print(f"Name: {run.name}, ID: {run.id}")


if __name__ == "__main__":
    fire.Fire({"main": main, "search_run": search_run, "fetch_run_by_url": fetch_run_by_url, "list_runs": list_runs})  # Add CLI command  # Add CLI command
