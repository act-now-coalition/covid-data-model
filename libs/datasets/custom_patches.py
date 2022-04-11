from typing import List
from datapublic.common_fields import CommonFields
from libs.datasets import AggregationLevel
from libs.datasets import timeseries
from libs.pipeline import RegionMask
import dataclasses
from libs.datasets.new_cases_and_deaths import spread_first_reported_value_after_stall

import pandas as pd

MD_COUNTIES = RegionMask(AggregationLevel.COUNTY, states=["MD"])
MD_STATE = RegionMask(AggregationLevel.STATE, states=["MD"])


def patch_maryland_missing_case_data(
    dataset: timeseries.MultiRegionDataset,
    start: str = "2021-12-04",
    end: str = "2021-12-29",
    locations: List[RegionMask] = [MD_STATE, MD_COUNTIES],
) -> timeseries.MultiRegionDataset:
    """Patch to fill in strings of days where the MD department of health did not report case data. 
    
    In December 2021, MD was hit with a cyber attack and could not report case data for most of the month.
    As such, the days are null. The R(t) metric failed due to the missing data, so this backfills
    the data reported after the cyber attack in order to revive the R(t) calculation.

    Enables the reporting stall backfill code to spread cases for stalls between the specified date range. 
    Overrides the default maximum backfill length of 7 days.
    """

    number_of_days = (pd.to_datetime(end) - pd.to_datetime(start)).days

    location_ds, other = dataset.partition_by_region(include=locations)
    dates_to_replace = list(pd.date_range(start=start, end=end))
    # Need to convert timestamp to strings to match the index dtype.
    dates_to_replace = [str(date.date()) for date in dates_to_replace]

    modified_ts = location_ds.timeseries_bucketed
    missing_dates_index = modified_ts.index.get_level_values("date").isin(dates_to_replace)

    # backfill the data within the date range and paste this data into the series.
    backfilled_data = spread_first_reported_value_after_stall(
        series=modified_ts[missing_dates_index][CommonFields.NEW_CASES],
        max_days_to_spread=number_of_days,
        is_multiindex=True,
    )
    modified_ts.loc[missing_dates_index, CommonFields.NEW_CASES] = backfilled_data

    subset_ds = dataclasses.replace(location_ds, timeseries_bucketed=modified_ts)
    return other.append_regions(subset_ds)
