import numpy as np
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import timeseries


def calculate_icu_capacity(region_df: pd.DataFrame):

    # TODO(chris): Why does the common fields break if it's not in the array?
    icu_beds = region_df.get(CommonFields.ICU_BEDS.value)
    current_total_icu = region_df.get(CommonFields.CURRENT_ICU_TOTAL.value)
    if icu_beds is None or current_total_icu is None:
        return np.nan

    icu_capacity = current_total_icu / icu_beds
    return icu_capacity


def calculate_all_icu_capacity_ratio(dataset: timeseries.MultiRegionDataset):

    dataset.groupby_region().apply(_calculate_icu_capacity)
