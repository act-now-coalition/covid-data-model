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
from libs.qa.dataset_summary_gen import generate_field_summary

IGNORE_COLUMNS = [
    CommonFields.STATE,
    CommonFields.COUNTRY,
    CommonFields.COUNTY,
    CommonFields.AGGREGATE_LEVEL,
]

VARIABLE_FIELD = "variable"


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
        [CommonFields.FIPS, CommonFields.DATE, VARIABLE_FIELD]
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


def get_summaries(
    sha1: str, sha2: str, level: Optional[AggregationLevel] = None, fips: Optional[str] = None
) -> Tuple[TimeseriesSummary, TimeseriesSummary]:
    """Builds summaries comparing timeseries between two commit shas.

    Args:
        sha1: First commit to compare.
        sha2: Second commit to compare.
        level: Optional AggregationLevel, if set will restrict comparisons to that level.
        fips: Optional fips to restrict summaries to.

    Returns: Tuple of summaries for each sha.
    """
    timeseries1 = combined_datasets.load_us_timeseries_dataset(commit=sha1)
    timeseries2 = combined_datasets.load_us_timeseries_dataset(commit=sha2)

    if level:
        timeseries1 = timeseries1.get_subset(aggregation_level=level)
        timeseries2 = timeseries2.get_subset(aggregation_level=level)

    if fips:
        timeseries1 = timeseries1.get_subset(fips=fips)
        timeseries2 = timeseries2.get_subset(fips=fips)

    summary1 = summarize_timeseries_fields(timeseries1.data)
    summary2 = summarize_timeseries_fields(timeseries2.data)

    sum1 = TimeseriesSummary(
        sha=sha1, timeseries=timeseries1, summary=summary1, fips=fips, level=level
    )
    sum2 = TimeseriesSummary(
        sha=sha2, timeseries=timeseries2, summary=summary2, fips=fips, level=level
    )

    return sum1, sum2


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
