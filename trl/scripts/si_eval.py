# %%
import os
import fire
import shortuuid
import json
import subprocess
from pathlib import Path
import pandas as pd
import tempfile
import logging
from orng import to_orng

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pkg_version(pkg_name: str) -> str:
    from importlib.metadata import version, PackageNotFoundError

    try:
        return version(pkg_name)
    except PackageNotFoundError:
        return None


def sync_dir(src: str, dst: str, parallel: int = 16) -> None:
    cmd = ["bbb", "sync", "--concurrency", str(parallel), f"{src}/", f"{dst}/"]
    os.system(" ".join(cmd))


class SpeechInsight:
    def __init__(self):
        self._tools_dir = Path("~/tools").expanduser()
        self._metrics_bin = self._tools_dir / "speechinsight_tools/linux-x64/framework-dependent/GetMetrics"
        self._setup()

    def _setup(self):
        REMOTE_TOOLS_DIR = to_orng("az://orngcresco/data/boren/data/tools")

        logger.info("Setting up SpeechInsight tools...")
        if not self._metrics_bin.exists():
            logger.info("Syncing SpeechInsight tools...")
            sync_dir(REMOTE_TOOLS_DIR, str(self._tools_dir))

        assert self._metrics_bin.exists(), f"Failed to find SpeechInsight tools to {self._tools_dir}"

        os.system(f"chmod +x {self._metrics_bin}")
        os.environ["LD_LIBRARY_PATH"] = f"{self._tools_dir}/speechinsight_tools/linux-x64/framework-dependent/:" + os.environ.get("LD_LIBRARY_PATH", "")

        if pkg_version("ter") is None:
            logger.info("Installing TER package...")
            os.system(f"pip install {str(self._tools_dir)}/ter-2.2.0-py3-none-any.whl")

        dotnet_dir = self._tools_dir / "dotnet"
        dotnet_bin = dotnet_dir / "dotnet"
        if not dotnet_bin.exists():
            dotnet_dir.mkdir(parents=True, exist_ok=True)
            os.system(f"tar -xzvf {self._tools_dir}/dotnet-runtime-8.0.0-linux-x64.tar.gz  -C {dotnet_dir}")

        os.environ["DOTNET_ROOT"] = str(dotnet_dir)
        os.environ["PATH"] = f"{dotnet_dir}:" + os.environ.get("PATH", "")

        logger.info("SpeechInsight setup completed.")
        logger.info("Installed .NET runtimes:")
        os.system("dotnet --list-runtimes")
        logger.info(f"Metrics binary: {self._metrics_bin}")
        logger.info(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

    def measure(self, df, metric="ewer", locale="en-US", tmp_dir=None):
        tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="si_tmp")
        logger.info(f"Using temporary directory for SI: {tmp_dir}")
        tsv_path = Path(tmp_dir) / f"{shortuuid.uuid()}.tsv"
        assert {"id", "hyp", "ref"} <= set(df.columns), "dfFrame must contain 'id', 'hyp', and 'ref' columns"
        df[["id", "hyp", "ref"]].to_csv(tsv_path, sep="\t", index=False, header=False)
        output_dir = Path(tmp_dir) / "output"

        if metric in ["wer", "displaywer"]:
            cmd = f"{self._metrics_bin} -d -t {tsv_path} -o {output_dir} -l {locale} --idcol 0 --recocol 1 --transcol 2"
        elif metric == "lexicalwer":
            cmd = f"{self._metrics_bin} -t {tsv_path} -o {output_dir} -l {locale} --idcol 0 --recocol 1 --transcol 2"
        elif metric == "ewer":
            cmd = f"{self._metrics_bin} -t {tsv_path} -o {output_dir} -l {locale} --idcol 0 --recocol 1 --transcol 2 --nondisfluency"
        elif metric == "ter":
            cmd = f"python speechinsight_tools/getdfmetrics.py ter -i {tsv_path} -o {output_dir} --locale {locale} --utt_id_idx 0 --disp_trans_idx 2 --disp_reco_idx 1 --ter_type nondisfluency"
        subprocess.run(cmd, capture_output=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stdin=subprocess.DEVNULL, shell=True, check=True)
        result = {
            **read_wer(output_dir),
            **read_ewer(output_dir),
        }
        return result


def read_wer(output_dir: Path):
    wer_file = output_dir / "WER.txt"
    assert wer_file.exists(), "WER.txt must exists"
    # WER 2.%
    wer_str = wer_file.read_text().split("\t")[1]
    wer = float(wer_str.strip("%"))
    return {"wer": wer}


def read_ewer(output_dir: Path):
    ewer_file = output_dir / "Entity_WER.txt"
    assert ewer_file.exists(), "Entity_WER.txt must exists"
    try:
        df = pd.read_csv(ewer_file, sep="\t", header=0)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    if df.empty:
        return {"ewer": None, "eer": None}
    df["ewer"] = df["EntityWer"].apply(lambda x: float(x.strip("%")))
    df["recall"] = df["#EntMatched"] / df["#Ent"] * 100.0
    df["eer"] = 100 - df["recall"]
    return {
        "ewer": float(df["ewer"].values[0]),
        "eer": float(df["eer"].values[0]),
    }


global_si = None


def get_speechinsight():
    global global_si
    if global_si is None:
        global_si = SpeechInsight()
    return global_si


def si_measure(df, metric="ewer", locale="en-US"):
    si = get_speechinsight()
    return si.measure(df, metric=metric, locale=locale)


def measure_result(result_file, metric: str = "ewer", locale: str = "en-US"):
    result_file = Path(result_file)
    stem = result_file.stem
    si_summary_path = result_file.parent / (stem.replace("_results", "_si_summary") + ".json")

    df = pd.read_json(result_file, lines=True)
    df.rename(columns={"ref": "Transcription"}, inplace=True)
    si_results = si_measure(df, metric=metric, locale=locale)
    with open(si_summary_path, "w") as f:
        json.dump(si_results, f)
    print(f"SI summary for {stem}: \n{si_results}")
    print(f"SI summary saved to {si_summary_path}")
    return si_results


def main(result_path: str, metric: str = "ewer", locale: str = "en-US"):
    """Main entry point for evaluating speech recognition results using SpeechInsight."""
    result_files = [result_path] if os.path.isfile(result_path) else list(Path(result_path).glob("*_results.jsonl"))
    for result_file in result_files:
        print(f"Measuring SI for {result_file}...")
        si_results = measure_result(result_file, metric=metric, locale=locale)
    return si_results


# %%

if __name__ == "__main__":
    fire.Fire(main)
