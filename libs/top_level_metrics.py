import numpy as np
import pandas as pd
from datetime import timedelta
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset
from libs import series_utils


# We will assume roughly 5 tracers are needed to trace a case within 48h.
# The range we give here could be between 5-15 contact tracers per case.
CONTACT_TRACERS_PER_CASE = 5


class MetricsFields:
    CASE_DENSITY_RATIO = "caseDensity"
    TEST_POSITIVITY = "testPositivityRatio"
    CONTACT_TRACER_CAPACITY_RATIO = "contactTracerCapacityRatio"


def calculate_top_level_metrics_for_fips(fips: str):
    timeseries = combined_datasets.load_us_timeseries_dataset()
    latest = combined_datasets.load_us_latest_dataset()

    fips_timeseries = timeseries.get_subset(fips=fips)
    fips_record = latest.get_record_for_fips(fips)

    # not sure of return type for now, could be a dictionary, or maybe it would be more effective
    # as a pandas dataframe with a column for each metric.
    return calculate_top_level_metrics_for_timeseries(fips_timeseries, fips_record)


def calculate_top_level_metrics_for_timeseries(
    timeseries: TimeseriesDataset, latest: dict
) -> pd.DataFrame:
    # Making sure that the timeseries object passed in is only for one fips.
    assert len(timeseries.all_fips) == 1
    fips = latest[CommonFields.FIPS]
    population = latest[CommonFields.POPULATION]

    data = timeseries.data

    cumulative_cases = series_utils.interpolate_stalled_values(data[CommonFields.CASES])
    case_density = calculate_case_density(cumulative_cases, population)

    cumulative_positive_tests = series_utils.interpolate_stalled_values(
        data[CommonFields.POSITIVE_TESTS]
    )
    cumulative_negative_tests = series_utils.interpolate_stalled_values(
        data[CommonFields.NEGATIVE_TESTS]
    )
    test_positivity = calculate_test_positivity(
        cumulative_positive_tests, cumulative_negative_tests
    )
    contact_tracer_capacity = calculate_contact_tracers(
        cumulative_cases, data[CommonFields.CONTACT_TRACERS_COUNT]
    )
    top_level_metrics_data = {
        CommonFields.FIPS: fips,
        CommonFields.DATE: data[CommonFields.DATE],
        MetricsFields.CASE_DENSITY_RATIO: case_density,
        MetricsFields.TEST_POSITIVITY: test_positivity,
        MetricsFields.CONTACT_TRACER_CAPACITY_RATIO: contact_tracer_capacity,
    }
    return pd.DataFrame(top_level_metrics_data, index=test_positivity.index)


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
    """Calculates positive test rate.

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

    if any(last_n_positive) and last_n_negative.isna().all():
        return pd.Series([], dtype="float64")
    return positive_smoothed / (negative_smoothed + positive_smoothed)


def calculate_contact_tracers(
    cases: pd.Series,
    contact_tracers: pd.Series,
    contact_tracers_per_case: int = CONTACT_TRACERS_PER_CASE,
) -> pd.Series:
    """Calculates ratio of hired tracers to estimated tracers needed based on daily cases.

    Args:
        cases: Cumulative cases.
        contact_tracers: Current tracers hired.
        contact_tracers_per_case: Number of tracers needed per case to effectively trace
            related cases within 48 hours.

    Returns: Series aligned on the same index as cases.
    """

    daily_cases = cases.diff()
    smoothed_daily_cases = series_utils.smooth_with_rolling_average(daily_cases)
    return contact_tracers / (smoothed_daily_cases * contact_tracers_per_case)


# Example of running calculation for all counties in a state, using the latest dataset
# to get all fips codes for that state
def calculate_metrics_for_counties_in_state(state: str):
    latest = combined_datasets.load_us_latest_dataset()
    state_latest_values = latest.county.get_subset(state=state)
    for fips in state_latest_values.all_fips:
        yield calculate_top_level_metrics_for_fips(fips)
