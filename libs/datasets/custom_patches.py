from datapublic.common_fields import CommonFields
from libs.datasets import AggregationLevel
from libs.datasets import timeseries
from libs.pipeline import RegionMask
import dataclasses

import pandas as pd

MD_COUNTIES = RegionMask(AggregationLevel.COUNTY, states=["MD"])
MD_STATE = RegionMask(AggregationLevel.STATE, states=["MD"])


def patch_maryland_missing_case_data(dataset: timeseries.MultiRegionDataset):
    """Patch to fill in strings of days where the MD department of health did not report case data. 
    
    In December 2021, MD was hit with a cyber attack and could not report case data for most of the month.
    As such, the days are null. The R(t) metric failed due to the missing data, so this fills in the data with
    zeros in order to revive the R(t) calculation.
    """

    # The state- and county-level data came back online on different dates
    # so we need to fill different date ranges for each.
    dataset = fill_zero_days(
        dataset=dataset, locations=MD_COUNTIES, start="2021-12-5", end="2021-12-27"
    )
    return fill_zero_days(dataset=dataset, locations=MD_STATE, start="2021-12-5", end="2021-12-19")


def fill_zero_days(
    dataset: timeseries.MultiRegionDataset, locations: RegionMask, start: str, end: str
):
    location_ds, other = dataset.partition_by_region(include=[locations])
    dates_to_replace = list(pd.date_range(start=start, end=end))

    modified_ts = location_ds.timeseries
    missing_dates_index = modified_ts.index.get_level_values("date").isin(dates_to_replace)

    modified_ts.loc[missing_dates_index, CommonFields.NEW_CASES] = 0

    md_state = dataclasses.replace(location_ds, timeseries=modified_ts, timeseries_bucketed=None)
    return other.append_regions(md_state)
