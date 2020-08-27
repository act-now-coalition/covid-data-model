from typing import Optional
import enum
from datetime import timedelta
import pandas as pd
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_fields

from api import can_api_definition
from libs import series_utils
from libs.datasets import combined_datasets
from libs.datasets import can_model_output_schema as schema
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs import icu_headroom_metric

Metrics = can_api_definition.Metrics

# We will assume roughly 5 tracers are needed to trace a case within 48h.
# The range we give here could be between 5-15 contact tracers per case.
CONTACT_TRACERS_PER_CASE = 5

#
RT_TRUNCATION_DAYS = 7


class MetricsFields(common_fields.ValueAsStrMixin, str, enum.Enum):
    # Note that the values of these fields must match the field names of the `Metrics`
    # class in `can_api_definition`
    CASE_DENSITY_RATIO = "caseDensity"
    TEST_POSITIVITY = "testPositivityRatio"
    CONTACT_TRACER_CAPACITY_RATIO = "contactTracerCapacityRatio"
    INFECTION_RATE = "infectionRate"
    INFECTION_RATE_CI90 = "infectionRateCI90"
    ICU_HEADROOM = "icuHeadroom"


def calculate_top_level_metrics_for_fips(fips: str):
    timeseries = combined_datasets.load_us_timeseries_dataset()
    latest = combined_datasets.load_us_latest_dataset()

    fips_timeseries = timeseries.get_subset(fips=fips)
    fips_record = latest.get_record_for_fips(fips)

    # not sure of return type for now, could be a dictionary, or maybe it would be more effective
    # as a pandas dataframe with a column for each metric.
    return calculate_metrics_for_timeseries(fips_timeseries, fips_record, None)


def calculate_metrics_for_timeseries(
    timeseries: TimeseriesDataset, latest: dict, model_output: Optional[CANPyseirLocationOutput]
) -> pd.DataFrame:
    # Making sure that the timeseries object passed in is only for one fips.
    assert len(timeseries.all_fips) == 1
    fips = latest[CommonFields.FIPS]
    population = latest[CommonFields.POPULATION]

    data = timeseries.data.set_index(CommonFields.DATE)

    infection_rate = np.nan
    infection_rate_ci90 = np.nan

    if model_output:
        # TODO(chris): Currently merging model output data into the timeseries data to align model
        # data with raw data.  However, if the index was properly set on both datasets to be DATE,
        # this would not be necessary.  In the future, consider indexing data on date so that
        # merges are not necessary.

        # Only merging date up to the most recent timeseries date (model data includes
        # future projections for other values and we don't want to pad the end with NaNs).
        up_to_latest_day = model_output.data[schema.DATE] <= data.index.max()
        fields_to_include = [
            schema.DATE,
            schema.RT_INDICATOR,
            schema.RT_INDICATOR_CI90,
            schema.CURRENT_ICU,
        ]
        model_data = model_output.data.loc[up_to_latest_day, fields_to_include]
        model_data = model_data.set_index(schema.DATE)

        infection_rate = model_data[schema.RT_INDICATOR]
        infection_rate_ci90 = model_data[schema.RT_INDICATOR_CI90]

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

    # Caculate icu headroom
    decomp = icu_headroom_metric.get_decomp_for_state(latest[CommonFields.STATE])
    icu_data = icu_headroom_metric.ICUMetricData(data, latest, decomp)
    icu_metric = icu_headroom_metric.calculate_icu_utilization_metric(icu_data)

    top_level_metrics_data = {
        CommonFields.FIPS: fips,
        MetricsFields.CASE_DENSITY_RATIO: case_density,
        MetricsFields.TEST_POSITIVITY: test_positivity,
        MetricsFields.CONTACT_TRACER_CAPACITY_RATIO: contact_tracer_capacity,
        MetricsFields.INFECTION_RATE: infection_rate,
        MetricsFields.INFECTION_RATE_CI90: infection_rate_ci90,
        MetricsFields.ICU_HEADROOM: icu_metric["metric"],
    }
    metrics = pd.DataFrame(top_level_metrics_data)
    return metrics.reset_index()


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


def calculate_latest_metrics(data: pd.DataFrame) -> Metrics:
    """Calculate latest metrics from top level metrics data.

    Args:
        data: Top level metrics timeseries data.

    Returns: Metrics
    """
    metrics = {}

    # Get latest value from data where available.
    for field in MetricsFields:
        last_available = data[field].last_valid_index()
        if last_available is None:
            metrics[field] = None
        else:
            metrics[field] = data[field][last_available]

    latest_rt = metrics[MetricsFields.INFECTION_RATE]

    if pd.isna(latest_rt):
        return Metrics(**metrics)

    # Infection rate is handled differently - the infection rate surfaced is actually the value
    # `RT_TRUNCATION_DAYS` in the past.
    data = data.set_index(CommonFields.DATE)
    last_rt_index = data[MetricsFields.INFECTION_RATE].last_valid_index()
    rt_index = last_rt_index + timedelta(days=-RT_TRUNCATION_DAYS)

    if rt_index not in data.index:
        metrics[MetricsFields.INFECTION_RATE] = None
        metrics[MetricsFields.INFECTION_RATE_CI90] = None
        return Metrics(**metrics)

    metrics[MetricsFields.INFECTION_RATE] = data[MetricsFields.INFECTION_RATE][rt_index]
    metrics[MetricsFields.INFECTION_RATE_CI90] = data[MetricsFields.INFECTION_RATE_CI90][rt_index]
    return Metrics(**metrics)
