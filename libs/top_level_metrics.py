import numpy as np
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset


def calculate_top_level_metrics_for_fips(fips: str):
    timeseries = combined_datasets.load_us_timeseries_dataset()
    latest = combined_datasets.load_us_latest_dataset()

    fips_timeseries = timeseries.get_subset(fips=fips)
    fips_record = latest.get_record_for_fips(fips)

    # not sure of return type for now, could be a dictionary, or maybe it would be more effective
    # as a pandas dataframe with a column for each metric.
    return calculate_top_level_metrics_for_timeseries(fips_timeseries, fips_record)


def calculate_top_level_metrics_for_timeseries(timeseries: TimeseriesDataset, latest: dict):
    # Making sure that the timeseries object passed in is only for one fips.
    assert len(timeseries.all_fips) == 1
    population = latest[CommonFields.POPULATION]
    neg_tests_cumulative = timeseries.data[CommonFields.NEGATIVE_TESTS]
    neg_tests_daily = neg_tests_cumulative.diff()

    pos_tests_cumulative = timeseries.data[CommonFields.POSITIVE_TESTS]
    pos_tests_daily = pos_tests_cumulative.diff()

    case_density = calculate_case_density(cases=pos_tests_daily, population=population)
    test_positivity = calculate_test_positivity(
        pos_cases=pos_tests_daily, neg_tests=neg_tests_daily
    )

    return {"case_density": case_density, "test_positivity": test_positivity}


def calculate_case_density(
    cases: pd.Series, population: int, smooth: int = 7, normalize_by: int = 100000
) -> pd.Series:
    """
    Calculates normalized cases density.

    Args:
        cases: Number of cases in a given fips.
        population: Population for a given fips.
        normalized_by: Normalize data by a constant.

    Returns:
        Population cases density.
    """
    smoothed = smoothWithRollingAverage(series=cases)
    return smoothed / (population / normalize_by)


def calculate_test_positivity(
    pos_cases: pd.Series, neg_tests: pd.Series, smooth: int = 7, lag_lookback: int = 7
) -> pd.Series:
    """
    Calculates positive test rate.

    Args:
        pos_cases: Number of positive cases.
        neg_tests: Number of negative cases.

    Returns:
        Positive test rate.
    """
    pos_smoothed = smoothWithRollingAverage(series=pos_cases)
    neg_smoothed = smoothWithRollingAverage(series=neg_tests, includeTrailingZeros=False)

    last_n_pos = pos_smoothed[-lag_lookback:]
    last_n_neg = neg_smoothed[-lag_lookback:]
    # TODO: Porting from: https://github.com/covid-projections/covid-projections/blob/master/src/common/models/Projection.ts#L521.
    # Do we still want to return no data if there appears to be positive case data but lagging data for negative cases?
    if any(last_n_pos) and last_n_neg.isna().all():
        return pd.Series([], dtype="float64")
    return pos_smoothed / (neg_smoothed + pos_smoothed)


# Example of running calculation for all counties in a state, using the latest dataset
# to get all fips codes for that state
def calculate_metrics_for_counties_in_state(state: str):
    latest = combined_datasets.load_us_latest_dataset()
    state_latest_values = latest.county.get_subset(state=state)
    for fips in state_latest_values.all_fips:
        yield calculate_top_level_metrics_for_fips(fips)


def smoothWithRollingAverage(series: pd.Series, window: int = 7, includeTrailingZeros: bool = True):
    """
    Smooths series with a min period of 1.

    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rolling.html

    Port of Projections.ts:
    https://github.com/covid-projections/covid-projections/blob/master/src/common/models/Projection.ts#L715

    Args:
        series: Series to smooth.
        window: Sliding window to average.
        includeTrailingZeros: Whether or not to NaN out trailing zeroes.

    Returns:
        Smoothed series.
    """
    rolling_average = series.rolling(window, min_periods=1).mean()
    if includeTrailingZeros:
        return rolling_average
    last_valid_index = series.replace(0, np.nan).last_valid_index()
    rolling_average[last_valid_index + 1 :] = np.nan
    return rolling_average
