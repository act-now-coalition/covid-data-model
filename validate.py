import argparse
from datetime import datetime
import os
import subprocess
import sys

from libs import build_params


def run_run_py(datasource: str) -> None:
    """
    Check that we successfully ran `python run.py`
    """
    env = os.environ.copy()
    env["COVID_DATA_PUBLIC"] = datasource
    start = datetime.now()
    # Will raise an appropriate exception if run.py crashed
    subprocess.run(
        ["python", "run.py"],
        cwd=os.getcwd(),
        # shell=True,
        env=env,
        capture_output=True,
        check=True,
    )
    end = datetime.now()
    duration = end - start
    print(f"run.py duration {duration}")


def clear_result_dir(result_dir: str) -> None:
    if os.path.exists(result_dir):
        for f in os.listdir(result_dir):
            os.unlink(os.path.join(result_dir, f))


UNSUPPORTED_REGIONS = ["AS", "GU", "MP"]


def validate_results(result_dir: str) -> None:
    """
    For each state, check that we have a file for each intervention,
    and that the file is non-empty
    """
    per_state_expected = len(build_params.get_interventions())
    missing_or_empty = []
    for state in build_params.US_STATE_ABBREV.values():
        if state in UNSUPPORTED_REGIONS:
            continue
        for i in range(per_state_expected):
            fname = os.path.join(result_dir, ".".join([state, str(i), "json"]))
            try:
                result = os.stat(fname)
                if result.st_size == 0:
                    missing_or_empty.append(fname)
            except FileNotFoundError:
                missing_or_empty.append(fname)
    if len(missing_or_empty) > 0:
        raise RuntimeError(
            f'Missing or empty expected files: {", ".join(missing_or_empty)}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate run.py against a dataset. For example, with a local checkout, try `python validate.py -d `pwd`/../covid-data-public`",
    )
    parser.add_argument(
        "-d",
        "--data-source",
        required=True,
        help="Path or URL to an instance of the covid-data-public repo",
    )

    args = parser.parse_args()
    clear_result_dir(build_params.OUTPUT_DIR)
    try:
        run_run_py(args.data_source)
    except subprocess.CalledProcessError as e:
        print(f"run.py failed with code {e.returncode}")
        print(e.stderr.decode("utf-8"))
        sys.exit(e.returncode)
    validate_results(build_params.OUTPUT_DIR)
