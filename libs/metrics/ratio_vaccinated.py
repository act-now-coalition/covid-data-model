import pandas as pd
from libs.datasets import timeseries
from covidactnow.datapublic.common_fields import CommonFields


def calculate_ratio_initiated(dataset: timeseries.OneRegionTimeseriesDataset) -> pd.Series:
    """Calculate ratio of population initiating vaccination.

    Args:
        dataset: Data for a single region.

    Returns: Series
    """
    population = dataset.latest[CommonFields.POPULATION]
    return dataset.date_indexed[CommonFields.VACCINATIONS_INITIATED] / population


def calculate_ratio_completed(dataset: timeseries.OneRegionTimeseriesDataset) -> pd.Series:
    """Calculate ratio of population completing vaccination.

    Args:
        dataset: Data for a single region.

    Returns: Series
    """
    population = dataset.latest[CommonFields.POPULATION]
    return dataset.date_indexed[CommonFields.VACCINATIONS_COMPLETED] / population
