import numpy as np
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields


def calculate_icu_capacity(region_df: pd.DataFrame):
    """Calculate ICU Capacity ratio.

    Args:
        region_df: Data for a specific region.

    Returns: np.nan if data missing or pd.Series of ICU capacity.
    """
    # TODO(chris): Why does the common fields break if it's not in the array?
    icu_beds = region_df.get(CommonFields.ICU_BEDS.value)
    current_total_icu = region_df.get(CommonFields.CURRENT_ICU_TOTAL.value)
    if icu_beds is None or current_total_icu is None:
        return np.nan

    icu_capacity = current_total_icu / icu_beds
    return icu_capacity
