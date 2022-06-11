import dataclasses
import pandas as pd

from libs.datasets import timeseries
from datapublic.common_fields import CommonFields
from libs.datasets.dataset_utils import AggregationLevel

MultiRegionDataset = timeseries.MultiRegionDataset


def _rolling_sum_7day(daily: pd.Series):
    assert CommonFields.DATE in daily.index.names
    return daily.rolling(7).sum()


def calculate_weekly_column_from_daily(
    dataset_in: MultiRegionDataset,
    level_to_replace: AggregationLevel,
    field_in: CommonFields,
    field_out: CommonFields,
) -> MultiRegionDataset:
    """Calculate weekly occurrences from daily data using a rolling 7-day sum.

    Args:
        dataset_in: The dataset to update.
        level_to_replace: Aggregation level to update data for.
        field_in: Column to sum into weekly data, should have units of new daily occurrences (e.g. new daily hospital admissions).
        field_out: Column to sum data into. Result will have units of new weekly occurrences (e.g. new weekly hospital admissions).  
    """

    subset_ds = dataset_in.get_subset(aggregation_level=level_to_replace)
    subset_ts = subset_ds.timeseries_bucketed.copy()  # copy to avoid SettingWithCopy warning.

    # If there's no data for field_in then just return the original dataset.
    if field_in not in subset_ts.columns or not subset_ts[field_in].any():
        return dataset_in

    # Make sure we do not have any pre-existing weekly data before overwriting the field_out column.
    # TODO PARALLEL: rework assert to handle if col doesnt exist
    # assert not subset_ts[field_out].any()
    subset_ts[field_out] = _rolling_sum_7day(subset_ts[field_in])

    new_ts = subset_ts.combine_first(dataset_in.timeseries_bucketed)
    dataset_out = dataclasses.replace(dataset_in, timeseries_bucketed=new_ts, timeseries=None)
    return dataset_out


def add_weekly_hospitalizations(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Calculates weekly new hospitalizations for state-level locations using a rolling 7-day sum."""
    return calculate_weekly_column_from_daily(
        dataset_in,
        level_to_replace=AggregationLevel.STATE,
        field_in=CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID,
        field_out=CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID,
    )
