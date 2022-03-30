import dataclasses
import pandas as pd

from datapublic.common_fields import CommonFields
from libs.datasets import timeseries
from libs.pipeline import Region

from tests import test_helpers

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

    state_ds = dataset_in.get_subset(aggregation_level=level_to_replace)
    state_ts = state_ds.timeseries_bucketed.copy()  # copy to avoid SettingWithCopy warning.

    # If there's no data for field_in then just return the original dataset.
    if not state_ts[field_in].first_valid_index():
        return dataset_in

    # Make sure we do not have any pre-existing weekly data before overwriting the field_out column.
    assert not state_ts[field_out].any()
    state_ts[field_out] = _rolling_sum_7day(state_ts[field_in])

    new_ts = state_ts.combine_first(dataset_in.timeseries_bucketed)
    dataset_out = dataclasses.replace(dataset_in, timeseries_bucketed=new_ts, timeseries=None)
    return dataset_out


def add_weekly_hospitalizations(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Updates weekly new hospitalizations for state-level locations using a rolling 7-day sum."""

    # Remove/move to test after done local testing.
    if not dataset_in:
        county_data = {
            CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID: [
                2,
                3,
                4,
                2,
                3,
                4,
                2,
                3,
                4,
                2,
                3,
                4,
                2,
            ]
        }
        data = {
            Region.from_fips("34001"): county_data,
            Region.from_cbsa_code("11100"): {
                CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID: [
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                ],
            },
            Region.from_state("MA"): {
                CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID: [
                    40,
                    50,
                    60,
                    50,
                    60,
                    40,
                    30,
                    40,
                    50,
                    70,
                    50,
                    40,
                ],
            },
        }
        dataset_in = test_helpers.build_dataset(data)

    return calculate_weekly_column_from_daily(
        dataset_in,
        level_to_replace=AggregationLevel.STATE,
        field_in=CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID,
        field_out=CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID,
    )
