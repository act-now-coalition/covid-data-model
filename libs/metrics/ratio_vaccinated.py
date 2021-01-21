import pandas as pd
from libs.datasets import timeseries
from covidactnow.datapublic.common_fields import CommonFields


def calculate_ratio_initiated(timeseries_df: timeseries.OneRegionTimeseriesDataset) -> pd.Series:
    """Calculate ratio of population initiating vaccination.

    Args:
        timeseries_df: Data for a single region.

    Returns: Series
    """
    population = timeseries_df.latest[CommonFields.POPULATION]
    return timeseries_df.date_indexed[CommonFields.VACCINATIONS_INITIATED] / population


def calculate_ratio_completed(timeseries_df: timeseries.OneRegionTimeseriesDataset) -> pd.Series:
    """Calculate ratio of population completing vaccination.

    Args:
        timeseries_df: Data for a single region.

    Returns: Series
    """
    population = timeseries_df.latest[CommonFields.POPULATION]
    return timeseries_df.date_indexed[CommonFields.VACCINATIONS_COMPLETED] / population
