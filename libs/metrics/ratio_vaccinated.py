import numpy as np
import pandas as pd
from libs.datasets import timeseries
from covidactnow.datapublic.common_fields import CommonFields


def calculate_ratio_initiated(timeseries_df: timeseries.OneRegionTimeseriesDataset):
    """Calculate  ratio.

    Args:
        region_df: Data for a specific region.

    Returns: np.nan if data missing or pd.Series of ICU capacity.
    """
    population = timeseries_df.latest[CommonFields.POPULATION]
    return timeseries_df.date_indexed[CommonFields.VACCINATIONS_COMPLETED] / population


def calculate_ratio_completed(timeseries_df: timeseries.OneRegionTimeseriesDataset):
    population = timeseries_df.latest[CommonFields.POPULATION]
    return timeseries_df.date_indexed[CommonFields.VACCINATIONS_COMPLETED] / population
