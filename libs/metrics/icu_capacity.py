import numpy as np
import pandas as pd

from datapublic.common_fields import CommonFields


def calculate_icu_capacity(region_df: pd.DataFrame):
    """Calculate ICU Capacity ratio.

    Args:
        region_df: Data for a specific region.

    Returns: np.nan if data missing or pd.Series of ICU capacity.
    """
    columns = region_df.columns
    if CommonFields.ICU_BEDS not in columns or CommonFields.CURRENT_ICU_TOTAL not in columns:
        return np.nan

    icu_beds = region_df[CommonFields.ICU_BEDS]
    current_total_icu = region_df[CommonFields.CURRENT_ICU_TOTAL]
    icu_capacity = current_total_icu / icu_beds
    return icu_capacity
