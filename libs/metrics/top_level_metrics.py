from typing import Optional, Tuple
import enum
from datetime import timedelta
import pandas as pd
import numpy as np
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_fields

from api import can_api_v2_definition
from api.can_api_v2_definition import TestPositivityRatioMethod, TestPositivityRatioDetails
from libs import series_utils
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.metrics import icu_headroom
from libs.metrics import icu_capacity

Metrics = can_api_v2_definition.Metrics
ICUHeadroomMetricDetails = can_api_v2_definition.ICUHeadroomMetricDetails
# We will assume roughly 5 tracers are needed to trace a case within 48h.
# The range we give here could be between 5-15 contact tracers per case.
CONTACT_TRACERS_PER_CASE = 5

#
RT_TRUNCATION_DAYS = 7


# CMS and HHS testing data can both lag by more than 7 days. Let's use it unless it's >2 weeks old.
# TODO(michael): Consider having different lookback values per metric, but this is fine for now.
MAX_METRIC_LOOKBACK_DAYS = 15


EMPTY_TS = pd.Series([], dtype="float64")


class MetricsFields(common_fields.ValueAsStrMixin, str, enum.Enum):
    # Note that the values of these fields must match the field names of the `Metrics`
    # class in `can_api_v2_definition`
    CASE_DENSITY_RATIO = "caseDensity"
    TEST_POSITIVITY = "testPositivityRatio"
    CONTACT_TRACER_CAPACITY_RATIO = "contactTracerCapacityRatio"
    INFECTION_RATE = "infectionRate"
    INFECTION_RATE_CI90 = "infectionRateCI90"
    ICU_HEADROOM_RATIO = "icuHeadroomRatio"
    ICU_CAPACITY_RATIO = "icuCapacityRatio"


def has_data_in_past_10_days(series: pd.Series) -> bool:
    return series_utils.has_recent_data(series, days_back=10, required_non_null_datapoints=1)


def calculate_metrics_for_timeseries(
    timeseries: OneRegionTimeseriesDataset,
    rt_data: Optional[OneRegionTimeseriesDataset],
    icu_data: Optional[OneRegionTimeseriesDataset],
    log,
    require_recent_icu_data: bool = True,
) -> Tuple[pd.DataFrame, Metrics]:
    # Making sure that the timeseries object passed in is only for one fips.
    assert timeseries.has_one_region()
    latest = timeseries.latest
    fips = timeseries.region.fips
    population = latest[CommonFields.POPULATION]

    data = timeseries.data.set_index(CommonFields.DATE)

    estimated_current_icu = None
    infection_rate = np.nan
    infection_rate_ci90 = np.nan
    if rt_data and not rt_data.empty:
        rt_data = rt_data.date_indexed
        infection_rate = rt_data["Rt_MAP_composite"]
        infection_rate_ci90 = rt_data["Rt_ci95_composite"] - rt_data["Rt_MAP_composite"]

    if icu_data and not icu_data.empty:
        icu_data = icu_data.date_indexed
        estimated_current_icu = icu_data[CommonFields.CURRENT_ICU]

    new_cases = data[CommonFields.NEW_CASES]
    case_density = calculate_case_density(new_cases, population)

    test_positivity, test_positivity_details = copy_test_positivity(timeseries, log)

    contact_tracer_capacity = calculate_contact_tracers(
        new_cases, data[CommonFields.CONTACT_TRACERS_COUNT]
    )

    # Caculate icu headroom
    decomp = icu_headroom.get_decomp_for_state(latest[CommonFields.STATE])
    icu_data = icu_headroom.ICUMetricData(
        data, estimated_current_icu, latest, decomp, require_recent_data=require_recent_icu_data
    )
    icu_metric, icu_metric_details = icu_headroom.calculate_icu_utilization_metric(icu_data)

    icu_capacity_ratio = icu_capacity.calculate_icu_capacity(data)

    top_level_metrics_data = {
        CommonFields.FIPS: fips,
        MetricsFields.CASE_DENSITY_RATIO: case_density,
        MetricsFields.TEST_POSITIVITY: test_positivity,
        MetricsFields.CONTACT_TRACER_CAPACITY_RATIO: contact_tracer_capacity,
        MetricsFields.INFECTION_RATE: infection_rate,
        MetricsFields.INFECTION_RATE_CI90: infection_rate_ci90,
        MetricsFields.ICU_HEADROOM_RATIO: icu_metric,
        MetricsFields.ICU_CAPACITY_RATIO: icu_capacity_ratio,
    }
    metrics = pd.DataFrame(top_level_metrics_data)
    metrics.index.name = CommonFields.DATE
    metrics = metrics.reset_index()

    metric_summary = None
    if not metrics.empty:
        metric_summary = calculate_latest_metrics(
            metrics, icu_metric_details, test_positivity_details
        )

    return metrics, metric_summary


def _lookup_test_positivity_method(
    positive_tests_provenance: Optional[str], negative_tests_provenance: Optional[str], log
) -> TestPositivityRatioMethod:
    method = None
    if positive_tests_provenance and positive_tests_provenance == negative_tests_provenance:
        method = TestPositivityRatioMethod.get(positive_tests_provenance)
    if method is None:
        log.debug(
            "Unable to find TestPositivityRatioMethod",
            positive_tests_provenance=positive_tests_provenance,
            negative_tests_provenance=negative_tests_provenance,
        )
        method = TestPositivityRatioMethod.OTHER
    return method


def copy_test_positivity(
    dataset_in: OneRegionTimeseriesDataset, log,
) -> Tuple[pd.Series, TestPositivityRatioDetails]:
    data = dataset_in.date_indexed
    test_positivity = common_df.get_timeseries(data, CommonFields.TEST_POSITIVITY, EMPTY_TS)
    provenance = dataset_in.provenance.get(CommonFields.TEST_POSITIVITY)
    method = TestPositivityRatioMethod.get(provenance)
    if method is None:
        if provenance is not None:
            log.warning("Unable to find TestPositivityRatioMethod", provenance=provenance)
        method = TestPositivityRatioMethod.OTHER
    return test_positivity, TestPositivityRatioDetails(source=method)


def _calculate_smoothed_daily_cases(new_cases: pd.Series, smooth: int = 7):

    if new_cases.first_valid_index() is None:
        return new_cases

    new_cases = new_cases.copy()

    # Front filling all cases with 0s.  We're assuming all regions are accurately
    # reporting the first day a new case occurs.  This will affect the first few cases
    # in a timeseries, because it's smoothing over a full period, rather than just the first
    # couple days of reported data.
    new_cases[: new_cases.first_valid_index() - timedelta(days=1)] = 0
    smoothed = series_utils.smooth_with_rolling_average(new_cases, window=smooth)

    return smoothed


def calculate_case_density(
    new_cases: pd.Series, population: int, smooth: int = 7, normalize_by: int = 100_000
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
    smoothed_daily_cases = _calculate_smoothed_daily_cases(new_cases, smooth=smooth)
    return smoothed_daily_cases / (population / normalize_by)


def calculate_contact_tracers(
    new_cases: pd.Series,
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
    smoothed_daily_cases = _calculate_smoothed_daily_cases(new_cases, smooth=7)
    contact_tracers_ratio = contact_tracers / (smoothed_daily_cases * contact_tracers_per_case)
    contact_tracers_ratio = contact_tracers_ratio.replace([-np.inf, np.inf], np.nan)
    return contact_tracers_ratio


def calculate_latest_metrics(
    data: pd.DataFrame,
    icu_metric_details: Optional[ICUHeadroomMetricDetails],
    test_positivity_method: Optional[TestPositivityRatioDetails],
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
    metrics = {
        "testPositivityRatioDetails": test_positivity_method,
        "icuHeadroomDetails": icu_metric_details,
    }
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
    return Metrics(**metrics)
