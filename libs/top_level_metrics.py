import numpy as np
import pandas as pd
from datetime import timedelta
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset
from libs import series_utils


class MetricsFields:
    CASE_DENSITY = "case_density"
    TEST_POSITIVITY = "test_positivity"


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
    fips = latest[CommonFields.FIPS]
    population = latest[CommonFields.POPULATION]

    data = timeseries.data

    case_density = calculate_case_density(data[CommonFields.CASES], population)

    cumulative_positive_tests = series_utils.interpolate_stalled_values(
        data[CommonFields.POSITIVE_TESTS]
    )
    cumulative_negative_tests = series_utils.interpolate_stalled_values(
        data[CommonFields.NEGATIVE_TESTS]
    )
    test_positivity = calculate_test_positivity(
        cumulative_positive_tests, cumulative_negative_tests
    )

    top_level_metrics_data = {
        MetricsFields.CASE_DENSITY: case_density,
        MetricsFields.TEST_POSITIVITY: test_positivity,
        CommonFields.FIPS: fips,
        CommonFields.DATE: data[CommonFields.DATE],
    }

    return pd.DataFrame(top_level_metrics_data).replace({np.nan: None}).set_index(CommonFields.DATE)


def calculate_case_density(
    cases: pd.Series, population: int, smooth: int = 7, normalize_by: int = 100_000
) -> pd.Series:
    """Calculates normalized daily case density.

    Args:
        cases: Cumulative cases.
        population: Population.
        smooth: days to smooth data.
        normalized_by: Normalize data by a constant.

    Returns:
        Population cases density.
    """
    cases_daily = cases.diff()
    smoothed = series_utils.smooth_with_rolling_average(cases_daily)
    return smoothed / (population / normalize_by)


def calculate_test_positivity(
    positive_tests: pd.Series, negative_tests: pd.Series, smooth: int = 7, lag_lookback: int = 7
) -> pd.Series:
    """
    Calculates positive test rate.

    Args:
        positive_tests: Number of cumulative positive tests.
        negative_tests: Number of cumulative negative tests.

    Returns:
        Positive test rate.
    """

    daily_negative_tests = negative_tests.diff()
    daily_positive_tests = positive_tests.diff()

    positive_smoothed = series_utils.smooth_with_rolling_average(daily_positive_tests)
    negative_smoothed = series_utils.smooth_with_rolling_average(
        daily_negative_tests, include_trailing_zeros=False
    )

    last_n_positive = positive_smoothed[-lag_lookback:]
    last_n_negative = negative_smoothed[-lag_lookback:]
    # TODO: Porting from: https://github.com/covid-projections/covid-projections/blob/master/src/common/models/Projection.ts#L521.
    # Do we still want to return no data if there appears to be positive case data but lagging data for negative cases?
    if any(last_n_positive) and last_n_negative.isna().all():
        return pd.Series([], dtype="float64")
    return positive_smoothed / (negative_smoothed + positive_smoothed)


# Example of running calculation for all counties in a state, using the latest dataset
# to get all fips codes for that state
def calculate_metrics_for_counties_in_state(state: str):
    latest = combined_datasets.load_us_latest_dataset()
    state_latest_values = latest.county.get_subset(state=state)
    for fips in state_latest_values.all_fips:
        yield calculate_top_level_metrics_for_fips(fips)
