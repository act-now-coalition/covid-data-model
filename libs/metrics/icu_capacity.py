import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import timeseries


def calculate_icu_capacity(region_df: pd.DataFrame):

    icu_beds = region_df.get(CommonFields.ICU_BEDS)
    current_total_icu = region_df.get(CommonFields.CURRENT_ICU_TOTAL)

    if icu_beds is None or current_total_icu is None:
        return None

    icu_capacity = current_total_icu / icu_beds
    return icu_capacity


def calculate_all_icu_capacity_ratio(dataset: timeseries.MultiRegionDataset):

    dataset.groupby_region().apply(_calculate_icu_capacity)
