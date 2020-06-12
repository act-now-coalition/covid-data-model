import logging
import pathlib
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np

import click
import structlog

from covidactnow.datapublic.common_df import write_df_as_csv, sort_common_field_columns
from libs import github_utils
from libs.datasets import combined_datasets

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
def save_combined_csv(csv_path_format):
    """Save the combined datasets DataFrame, cleaned up for easier comparisons."""
    csv_path = csv_path_format.format(
        git_sha=subprocess.check_output(
            ["git", "describe", "--dirty", "--always", "--long"], text=True
        ).strip(),
        git_branch=subprocess.check_output(
            ["git", "symbolic-ref", "--short", "HEAD"], text=True
        ).strip(),
        timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"),
    )

    timeseries = combined_datasets.build_us_timeseries_with_all_fields()
    timeseries_data = timeseries.data

    timeseries_data = cleanup_df(timeseries_data)
    timeseries_data.info()

    write_df_as_csv(timeseries_data, csv_path, structlog.get_logger())


def cleanup_df(df):
    col_to_drop = [c for c in ["index"] if c in df.columns]
    # Replace experimental NA object with conventional `nan`. Without this `convert_dtypes` leaves the
    # negative_tests column with dtype=object and it is output as a mix of floats and ints, which makes
    # diff-ing harder.
    df = df.replace({pd.NA: np.nan})
    df = df.drop(columns=col_to_drop).convert_dtypes()
    # Make sure that columns that can be represented as an integer are converted to `Int64`. `int64` won't work
    # because many of our columns have missing values.
    for col in df.select_dtypes(include="number").columns:
        as_ints = df[col].astype("Int64", copy=False)
        error = (df[col] - as_ints).abs().sum()
        if error < 0.001:
            df[col] = as_ints
    return sort_common_field_columns(df)
