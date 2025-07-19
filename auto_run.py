import subprocess
import fire

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
