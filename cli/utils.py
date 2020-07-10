import logging
import pathlib
import subprocess
from datetime import datetime

import click
import structlog

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs import github_utils
from libs.datasets import combined_datasets
import pandas as pd
import numpy as np

from libs.qa.common_df_diff import DatasetDiff

_logger = logging.getLogger(__name__)


@click.group("utils")
def main():
    pass


@main.command()
@click.argument("run-number", type=int, required=False)
@click.option(
    "--github-token",
    envvar="GITHUB_TOKEN",
    required=True,
    help="Github Token, can be an option or set as env variable GITHUB_TOKEN",
)
@click.option("--output-dir", "-o", type=pathlib.Path, default=pathlib.Path("."))
def download_model_artifact(github_token, run_number, output_dir):
    """Download model output from github action publish and deploy workflow. """
    github_utils.download_model_artifact(github_token, output_dir, run_number=run_number)


@main.command()
@click.option(
    "--csv-path-format",
    default="combined-{git_branch}-{git_sha}-{timestamp}.csv",
    show_default=True,
    help="Filename template where CSV is written",
)
@click.option("--output-dir", "-o", type=pathlib.Path, default=pathlib.Path("."))
def save_combined_csv(csv_path_format, output_dir):
    """Save the combined datasets DataFrame, cleaned up for easier comparisons."""
    csv_path = form_path_name(csv_path_format, output_dir)

    timeseries = combined_datasets.build_us_timeseries_with_all_fields()
    timeseries_data = timeseries.data

    common_df.write_csv(timeseries_data, csv_path, structlog.get_logger())


@main.command()
@click.option(
    "--csv-path-format",
    default="latest-{git_branch}-{git_sha}-{timestamp}.csv",
    show_default=True,
    help="Filename template where CSV is written",
)
@click.option("--output-dir", "-o", type=pathlib.Path, default=pathlib.Path("."))
def save_combined_latest_csv(csv_path_format, output_dir):
    """Save the combined datasets latest DataFrame, cleaned up for easier comparisons."""
    csv_path = form_path_name(csv_path_format, output_dir)

    latest = combined_datasets.build_us_latest_with_all_fields()
    # This is a hacky modification of common_df.write_csv because it requires a date index.
    latest_data = latest.data.set_index(CommonFields.FIPS).replace({pd.NA: np.nan}).convert_dtypes()
    latest_data.to_csv(csv_path, date_format="%Y-%m-%d", index=True, float_format="%.12g")


def form_path_name(csv_path_format, output_dir):
    """Create a path from a format string that may contain `{git_sha}` etc and output_dir."""
    try:
        git_branch = subprocess.check_output(
            ["git", "symbolic-ref", "--short", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        git_branch = "no-HEAD-branch"
    csv_path = pathlib.Path(output_dir) / csv_path_format.format(
        git_sha=subprocess.check_output(
            ["git", "describe", "--dirty", "--always", "--long"], text=True
        ).strip(),
        git_branch=git_branch,
        timestamp=datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
    )
    return csv_path


@main.command()
@click.argument("csv_path_left", type=str, required=True)
@click.argument("csv_path_right", type=str, required=True)
def csv_diff(csv_path_left, csv_path_right):
    """Compare 2 CSV files."""
    df_l = common_df.read_csv(csv_path_left)
    df_r = common_df.read_csv(csv_path_right)

    differ_l = DatasetDiff.make(df_l)
    differ_r = DatasetDiff.make(df_r)
    differ_l.compare(differ_r)

    print(f"File: {csv_path_left}")
    print(differ_l)
    print(f"File: {csv_path_right}")
    print(differ_r)
