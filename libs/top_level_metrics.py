from typing import Optional, Tuple
import enum
from datetime import timedelta
import pandas as pd
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_fields

from api import can_api_definition
from libs import series_utils
from libs.datasets import can_model_output_schema as schema
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs import icu_headroom_metric

Metrics = can_api_definition.Metrics
ICUHeadroomMetricDetails = can_api_definition.ICUHeadroomMetricDetails
# We will assume roughly 5 tracers are needed to trace a case within 48h.
# The range we give here could be between 5-15 contact tracers per case.
CONTACT_TRACERS_PER_CASE = 5

#
RT_TRUNCATION_DAYS = 7


MAX_METRIC_LOOKBACK_DAYS = 7


class MetricsFields(common_fields.ValueAsStrMixin, str, enum.Enum):
    # Note that the values of these fields must match the field names of the `Metrics`
    # class in `can_api_definition`
    CASE_DENSITY_RATIO = "caseDensity"
    TEST_POSITIVITY = "testPositivityRatio"
    CONTACT_TRACER_CAPACITY_RATIO = "contactTracerCapacityRatio"
    INFECTION_RATE = "infectionRate"
    INFECTION_RATE_CI90 = "infectionRateCI90"
    ICU_HEADROOM_RATIO = "icuHeadroomRatio"


def calculate_metrics_for_timeseries(
    timeseries: OneRegionTimeseriesDataset,
    rt_data: Optional[OneRegionTimeseriesDataset],
    icu_data: Optional[OneRegionTimeseriesDataset],
    require_recent_icu_data: bool = True,
) -> Tuple[pd.DataFrame, Metrics]:
    # Making sure that the timeseries object passed in is only for one fips.
    assert timeseries.has_one_region()
    latest = timeseries.latest
    fips = latest[CommonFields.FIPS]
    population = latest[CommonFields.POPULATION]

    data = timeseries.data.set_index(CommonFields.DATE)

    estimated_current_icu = None
    infection_rate = np.nan
    infection_rate_ci90 = np.nan
    if rt_data and not rt_data.empty:
        rt_data = rt_data.date_indexed
        infection_rate = rt_data["Rt_MAP_composite"]
        infection_rate_ci90 = rt_data["Rt_ci95_composite"] - rt_data["Rt_MAP_composite"]

    if icu_data and not rt_data.empty:
        icu_data = icu_data.date_indexed
        estimated_current_icu = icu_data[CommonFields.CURRENT_ICU]

    cumulative_cases = data[CommonFields.CASES]
    case_density = calculate_case_density(cumulative_cases, population)

    cumulative_positive_tests = series_utils.interpolate_stalled_and_missing_values(
        data[CommonFields.POSITIVE_TESTS]
    )
    cumulative_negative_tests = series_utils.interpolate_stalled_and_missing_values(
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
    icu_data = icu_headroom_metric.ICUMetricData(
        data, estimated_current_icu, latest, decomp, require_recent_data=require_recent_icu_data
    )
    icu_metric, icu_metric_details = icu_headroom_metric.calculate_icu_utilization_metric(icu_data)

    top_level_metrics_data = {
        CommonFields.FIPS: fips,
        MetricsFields.CASE_DENSITY_RATIO: case_density,
        MetricsFields.TEST_POSITIVITY: test_positivity,
        MetricsFields.CONTACT_TRACER_CAPACITY_RATIO: contact_tracer_capacity,
        MetricsFields.INFECTION_RATE: infection_rate,
        MetricsFields.INFECTION_RATE_CI90: infection_rate_ci90,
        MetricsFields.ICU_HEADROOM_RATIO: icu_metric,
    }
    metrics = pd.DataFrame(top_level_metrics_data)
    metrics.index.name = CommonFields.DATE
    metrics = metrics.reset_index()

    metric_summary = None
    if not metrics.empty:
        metric_summary = calculate_latest_metrics(metrics, icu_metric_details)

    return metrics, metric_summary


def _calculate_smoothed_daily_cases(cases: pd.Series, smooth: int = 7):

    if cases.first_valid_index() is None:
        return cases

    cases = cases.copy()

    # Front filling all cases with 0s.  We're assuming all regions are accurately
    # reporting the first day a new case occurs.  This will affect the first few cases
    # in a timeseries, because it's smoothing over a full period, rather than just the first
    # couple days of reported data.
    cases[: cases.first_valid_index() - timedelta(days=1)] = 0
    cases_daily = cases.diff()
    smoothed = series_utils.smooth_with_rolling_average(cases_daily, window=smooth)

    return smoothed


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
    smoothed_daily_cases = _calculate_smoothed_daily_cases(cases, smooth=smooth)
    return smoothed_daily_cases / (population / normalize_by)


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

    smoothed_daily_cases = _calculate_smoothed_daily_cases(cases, smooth=7)
    return contact_tracers / (smoothed_daily_cases * contact_tracers_per_case)


def calculate_latest_metrics(
    data: pd.DataFrame,
    icu_metric_details: Optional[ICUHeadroomMetricDetails],
    max_lookback_days: int = MAX_METRIC_LOOKBACK_DAYS,
) -> Metrics:
    """Calculate latest metrics from top level metrics data.

    Args:
        data: Top level metrics timeseries data.
        icu_metric_details: Optional details about the icu headroom metric.
        max_lookback_days: Number of days back from the latest day to consider metrics.

    Returns: Metrics
    """
    data = data.set_index(CommonFields.DATE)
    metrics = {}
    latest_date = data.index[-1]

    # Get latest value from data where available.
    for field in MetricsFields:
        last_available = data[field].last_valid_index()
        if last_available is None:
            metrics[field] = None
        # Limiting metrics surfaced to be metrics updated in the last `max_lookback_days` of
        # data.
        elif last_available <= latest_date - timedelta(days=max_lookback_days):
            metrics[field] = None
        else:
            metrics[field] = data[field][last_available]

    if not data[MetricsFields.INFECTION_RATE].any():
        return Metrics(**metrics)

    # Infection rate is handled differently - the infection rate surfaced is actually the value
    # `RT_TRUNCATION_DAYS` in the past.
    last_rt_index = data[MetricsFields.INFECTION_RATE].last_valid_index()
    rt_index = last_rt_index + timedelta(days=-RT_TRUNCATION_DAYS)
    stale_rt = last_rt_index <= latest_date - timedelta(days=max_lookback_days)

    if stale_rt or rt_index not in data.index:
        metrics[MetricsFields.INFECTION_RATE] = None
        metrics[MetricsFields.INFECTION_RATE_CI90] = None
        return Metrics(**metrics)

    metrics[MetricsFields.INFECTION_RATE] = data[MetricsFields.INFECTION_RATE][rt_index]
    metrics[MetricsFields.INFECTION_RATE_CI90] = data[MetricsFields.INFECTION_RATE_CI90][rt_index]
    return Metrics(**metrics, icuHeadroomDetails=icu_metric_details)
