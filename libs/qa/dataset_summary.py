from typing import Optional, Tuple

import io
import pathlib
import pydantic
import pandas as pd
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import combined_datasets
from libs import git_lfs_object_helpers
from libs.qa.dataset_summary_gen import generate_field_summary


IGNORE_COLUMNS = [
    CommonFields.STATE,
    CommonFields.COUNTRY,
    CommonFields.COUNTY,
    CommonFields.AGGREGATE_LEVEL,
    CommonFields.LOCATION_ID,
]

VARIABLE_FIELD = "variable"

SUMMARY_PATH = dataset_utils.DATA_DIRECTORY / "timeseries_summary.csv"


class TimeseriesSummary(pydantic.BaseModel):
    """Summary of timeseries dataset at a given commit sha."""

    sha: str
    timeseries: TimeseriesDataset
    summary: pd.DataFrame
    fips: Optional[str]
    level: Optional[AggregationLevel]

    class Config:
        arbitrary_types_allowed = True


def summarize_timeseries_fields(data: pd.DataFrame) -> pd.DataFrame:
    data = data[[column for column in data.columns if column not in IGNORE_COLUMNS]]

    melted = pd.melt(data, id_vars=[CommonFields.FIPS, CommonFields.DATE]).set_index(
        CommonFields.DATE
    )
    fips_variable_grouped = melted.groupby([CommonFields.FIPS, VARIABLE_FIELD])
    return fips_variable_grouped["value"].apply(generate_field_summary).unstack()


def find_missing_values_in_summary(
    summary_l: pd.DataFrame, summary_r: pd.DataFrame
) -> pd.DataFrame:
    """Returns values that are in the left summary but not in right summary."""
    summary_l_value_index = summary_l[summary_l.has_value].index
    summary_r_missing = summary_r[summary_r.has_value == False]
    newly_missing_r = summary_r_missing[summary_r_missing.index.isin(summary_l_value_index)]
    return summary_l.loc[newly_missing_r.index]


def get_changes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Returns differences of each column in input DataFrames for matching indices.

    Expects dataframes with matching index levels.
    """
    if not (df1.index == df2.index).all():
        raise ValueError("Indexes must match")

    ne_stacked = (df1 != df2).stack()
    changed = ne_stacked[ne_stacked]
    difference_locations = np.where(df1 != df2)
    changed_from = df1.values[difference_locations]
    changed_to = df2.values[difference_locations]
    return pd.DataFrame({"from": changed_from, "to": changed_to}, index=changed.index)


def load_summary(path: pathlib.Path = SUMMARY_PATH, commit: Optional[str] = None) -> pd.DataFrame:

    if commit:
        data = git_lfs_object_helpers.get_data_for_path(path, commit=commit)
    else:
        data = path.read_bytes()

    buf = io.BytesIO(data)
    return pd.read_csv(buf, dtype={CommonFields.FIPS: str}).set_index(
        [CommonFields.FIPS, VARIABLE_FIELD]
    )
