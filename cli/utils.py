import logging
import pathlib
import subprocess
from datetime import datetime
from typing import Optional

import click
import structlog

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields, COMMON_FIELDS_TIMESERIES_KEYS
from libs import github_utils
from libs.datasets import combined_datasets
from pydantic import BaseModel
import pandas as pd
import numpy as np


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
        timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"),
    )
    return csv_path


class DatasetDiff(BaseModel):
    duplicates_dropped: pd.DataFrame
    melt: pd.DataFrame
    all_variable_fips: pd.MultiIndex
    my_ts: Optional[pd.MultiIndex] = None
    common_fips: Optional[pd.DataFrame] = None
    my_ts_points: Optional[pd.DataFrame] = None
    ts_diffs: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True

    def __str__(self):
        return f"""Duplicate rows in this file: {self.duplicates_dropped}
TS only in this file: {self.my_ts}
TS points only in this file: {self.my_ts_points.groupby('date').size().to_dict()}
TS diffs: {self.ts_diffs if self.ts_diffs is not None else ''}
TS diffs: {self.ts_diffs.groupby('variable has_overlap'.split()).mean() if self.ts_diffs is not None else ''}
"""

    @staticmethod
    def make(df: pd.DataFrame) -> "DatasetDiff":
        dups = df.loc[df.index.duplicated(keep=False)]
        if not dups.empty:
            df = df.drop_duplicates()

        df = df.reset_index()
        columns_to_drop = {
            "index",
            CommonFields.STATE,
            CommonFields.COUNTRY,
            CommonFields.AGGREGATE_LEVEL,
        }.intersection(df.columns)
        for col in df.select_dtypes(include="object"):
            if col != "fips":
                print(f"dropping based on type {col}")
                columns_to_drop.add(col)
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        melt = (
            df.melt(id_vars=COMMON_FIELDS_TIMESERIES_KEYS)
            .set_index(["variable"] + COMMON_FIELDS_TIMESERIES_KEYS)
            .dropna()
        )

        all_variable_fips = melt.groupby("variable fips".split()).first().index
        return DatasetDiff(duplicates_dropped=dups, melt=melt, all_variable_fips=all_variable_fips)

    def compare(self, other: "DatasetDiff"):
        # Index of <variable, fips> that have at least one real value in only one dataset
        self.my_ts = self.all_variable_fips.difference(other.all_variable_fips)
        other.my_ts = other.all_variable_fips.difference(self.all_variable_fips)

        common_variable_fips = self.all_variable_fips.intersection(other.all_variable_fips)
        self.common_fips = self.melt.loc[
            self.melt.reset_index(CommonFields.DATE).index.isin(common_variable_fips)
        ]
        other.common_fips = other.melt.loc[
            other.melt.reset_index(CommonFields.DATE).index.isin(common_variable_fips)
        ]

        joined_ts = pd.merge(
            self.common_fips["value"],
            other.common_fips["value"],
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("_l", "_r"),
        )
        joined_ts_notna = joined_ts.notna()
        self.my_ts_points = joined_ts.loc[
            joined_ts_notna["value_l"] & ~joined_ts_notna["value_r"], "value_r"
        ]
        other.my_ts_points = joined_ts.loc[
            joined_ts_notna["value_r"] & ~joined_ts_notna["value_l"], "value_l"
        ]
        self.ts_diffs = joined_ts.groupby("variable fips".split()).apply(timeseries_diff)


def timeseries_diff(ts: pd.DataFrame) -> float:
    try:
        ts = ts.droplevel(["variable", CommonFields.FIPS])
        right = ts["value_r"]
        left = ts["value_l"]
        start = max(right.idxmin(), left.idxmin())
        end = min(right.idxmax(), left.idxmax())
        if start <= end:
            right_common_ts = right.loc[start:end].interpolate(method="time")
            left_common_ts = left.loc[start:end].interpolate(method="time")
            diff = (
                (right_common_ts - left_common_ts).abs() / ((right_common_ts + left_common_ts) / 2)
            ).mean()
            if diff > 0.01:
                print(ts)
                print(right_common_ts)
                print(left_common_ts)
            return pd.Series(
                [diff, len(right_common_ts), True], index=["diff", "points_overlap", "has_overlap"]
            )
        else:
            return pd.Series([1.0, 0, False], index=["diff", "points_overlap", "has_overlap"])
    except:
        ts.info()
        print(ts)
        return float("NaN")


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
