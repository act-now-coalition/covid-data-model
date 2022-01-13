from typing import Optional, Tuple
import enum
from datetime import timedelta

import more_itertools
import pandas as pd
import numpy as np
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_fields

from api import can_api_v2_definition
from api.can_api_v2_definition import TestPositivityRatioMethod, TestPositivityRatioDetails
from libs import series_utils
from libs.datasets.new_cases_and_deaths import spread_first_reported_value_after_stall
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.metrics import icu_capacity

Metrics = can_api_v2_definition.Metrics
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
    ICU_CAPACITY_RATIO = "icuCapacityRatio"
    VACCINATIONS_INITIATED_RATIO = "vaccinationsInitiatedRatio"
    VACCINATIONS_COMPLETED_RATIO = "vaccinationsCompletedRatio"
    VACCINATIONS_ADDITIONAL_DOSE_RATIO = "vaccinationsAdditionalDoseRatio"


# These precisions should be inline with
# https://github.com/covid-projections/covid-projections/blob/c076f39f54dcf6ca3f20fbb67839c37bb8b0f5bf/src/common/metric.tsx#L90
# but note that percentages (ICU_*, TEST_POSITIVITY, VACCINATIONS_*) need an
# additional 2 digits of precisions since they will be converted from ratio
# (0.xyz) to percentage (XY.Z%).

METRIC_ROUNDING_PRECISION = {
    MetricsFields.CASE_DENSITY_RATIO: 1,
    MetricsFields.TEST_POSITIVITY: 3,
    MetricsFields.CONTACT_TRACER_CAPACITY_RATIO: 2,
    MetricsFields.INFECTION_RATE: 2,
    MetricsFields.INFECTION_RATE_CI90: 2,
    MetricsFields.ICU_CAPACITY_RATIO: 2,
    MetricsFields.VACCINATIONS_INITIATED_RATIO: 3,
    MetricsFields.VACCINATIONS_COMPLETED_RATIO: 3,
    MetricsFields.VACCINATIONS_ADDITIONAL_DOSE_RATIO: 3,
}


def has_data_in_past_10_days(series: pd.Series) -> bool:
    return series_utils.has_recent_data(series, days_back=10, required_non_null_datapoints=1)


def calculate_metrics_for_timeseries(
    timeseries: OneRegionTimeseriesDataset, rt_data: Optional[OneRegionTimeseriesDataset], log,
) -> Tuple[pd.DataFrame, Metrics]:
    # Making sure that the timeseries object passed in is only for one fips.
    assert timeseries.has_one_region()
    latest = timeseries.latest
    fips = timeseries.region.fips
    population = latest[CommonFields.POPULATION]

    data = timeseries.data.set_index(CommonFields.DATE)

    infection_rate = np.nan
    infection_rate_ci90 = np.nan
    if rt_data and not rt_data.empty:
        rt_data = rt_data.date_indexed
        infection_rate = rt_data["Rt_MAP_composite"]
        infection_rate_ci90 = rt_data["Rt_ci95_composite"] - rt_data["Rt_MAP_composite"]

    new_cases = data[CommonFields.NEW_CASES]
    case_density = calculate_case_density(new_cases, population)

    test_positivity, test_positivity_details = copy_test_positivity(timeseries, log)

    contact_tracer_capacity = calculate_contact_tracers(
        new_cases, data[CommonFields.CONTACT_TRACERS_COUNT]
    )

    icu_capacity_ratio = icu_capacity.calculate_icu_capacity(data)

    vaccines_initiated_ratio = (
        common_df.get_timeseries(
            timeseries.date_indexed, CommonFields.VACCINATIONS_INITIATED_PCT, EMPTY_TS
        )
        / 100.0
    )
    vaccines_completed_ratio = (
        common_df.get_timeseries(
            timeseries.date_indexed, CommonFields.VACCINATIONS_COMPLETED_PCT, EMPTY_TS
        )
        / 100.0
    )

    vaccines_additional_dose_ratio = (
        common_df.get_timeseries(
            timeseries.date_indexed, CommonFields.VACCINATIONS_ADDITIONAL_DOSE_PCT, EMPTY_TS
        )
        / 100.0
    )

    top_level_metrics_data = {
        CommonFields.FIPS: fips,
        MetricsFields.CASE_DENSITY_RATIO: case_density,
        MetricsFields.TEST_POSITIVITY: test_positivity,
        MetricsFields.CONTACT_TRACER_CAPACITY_RATIO: contact_tracer_capacity,
        MetricsFields.INFECTION_RATE: infection_rate,
        MetricsFields.INFECTION_RATE_CI90: infection_rate_ci90,
        MetricsFields.ICU_CAPACITY_RATIO: icu_capacity_ratio,
        MetricsFields.VACCINATIONS_INITIATED_RATIO: vaccines_initiated_ratio,
        MetricsFields.VACCINATIONS_COMPLETED_RATIO: vaccines_completed_ratio,
        MetricsFields.VACCINATIONS_ADDITIONAL_DOSE_RATIO: vaccines_additional_dose_ratio,
    }
    metrics = pd.DataFrame(top_level_metrics_data)
    metrics = metrics.round(METRIC_ROUNDING_PRECISION)
    metrics.index.name = CommonFields.DATE
    metrics = metrics.reset_index()

    metric_summary = None
    if not metrics.empty:
        metric_summary = calculate_latest_metrics(metrics, test_positivity_details)

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


def _remove_trailing_zeros_until_threshold(series: pd.Series, stall_length: int) -> pd.Series:

    series = pd.Series(series.values.copy(), index=series.index.get_level_values(CommonFields.DATE))
    last_nonzero_index = series.loc[series != 0].last_valid_index()
    last_index = series.last_valid_index()

    if last_nonzero_index is None:
        return series

    # If data has been zero for at least stall_length days then
    # we consider the data reported to be actual zeros instead of a reporting stall.
    # When this is the case we do not want to remove the trailing zeros.
    if (last_index - last_nonzero_index) >= pd.to_timedelta(stall_length, unit="day"):
        return series

    series[last_nonzero_index + pd.DateOffset(1) :] = None
    return series


def copy_test_positivity(
    dataset_in: OneRegionTimeseriesDataset, log,
) -> Tuple[pd.Series, TestPositivityRatioDetails]:
    data = dataset_in.date_indexed
    test_positivity = common_df.get_timeseries(data, CommonFields.TEST_POSITIVITY, EMPTY_TS)
    # Make a set to eliminate duplicates.
    provenance = set(dataset_in.provenance.get(CommonFields.TEST_POSITIVITY, []))
    method = None
    if len(provenance) == 1:
        method = TestPositivityRatioMethod.get(more_itertools.first(provenance))

    if method is None:
        # Most likely there were zero provenance or the one wasn't found. Less likely, there were
        # more than one unique values in provenance.
        method = TestPositivityRatioMethod.OTHER
        if provenance:
            log.warning("Unable to find TestPositivityRatioMethod", provenance=provenance)
    return test_positivity, TestPositivityRatioDetails(source=method)


def _calculate_smoothed_daily_cases(new_cases: pd.Series, smooth: int = 7, stall_length: int = 10):

    if new_cases.first_valid_index() is None:
        return new_cases

    new_cases = new_cases.copy()
    # NOTE(sean) 12/15/2021: When spread_first_reported_value_after_stall is applied,
    # the timeseries for locations with non-daily (e.g. weekly) reporting cadences are
    # pulled towards zero in between reporting days. This is because the backfilling removes some of the
    # weekly cases out of the 7-day window. To combat this, we remove trailing zeros from the data.

    # After a certain number of days (10 by default) we consider trailing
    # zeros to be real data and not a reporting lag.
    # After this threshold we no longer remove the trailing zeros.
    new_cases = _remove_trailing_zeros_until_threshold(new_cases, stall_length)

    # Front filling all cases with 0s.  We're assuming all regions are accurately
    # reporting the first day a new case occurs.  This will affect the first few cases
    # in a timeseries, because it's smoothing over a full period, rather than just the first
    # couple days of reported data.
    new_cases[: new_cases.first_valid_index() - timedelta(days=1)] = 0
    smoothed = series_utils.smooth_with_rolling_average(new_cases, window=smooth)

    return smoothed


def calculate_case_density(
    new_cases: pd.Series,
    population: int,
    smooth: int = 7,
    normalize_by: int = 100_000,
    stall_length: int = 10,
) -> pd.Series:
    """Calculates normalized daily case density.

    Args:
        cases: Cumulative cases.
        population: Population.
        smooth: days to smooth data.
        normalized_by: Normalize data by a constant.
        stall_length: Days worth of trailing zeros to truncate.

    Returns:
        Population cases density.
    """

    spread_cases = spread_first_reported_value_after_stall(new_cases)
    smoothed_spread_daily_cases = _calculate_smoothed_daily_cases(
        spread_cases, smooth=smooth, stall_length=stall_length
    )
    return smoothed_spread_daily_cases / (population / normalize_by)


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
    test_positivity_method: Optional[TestPositivityRatioDetails],
    max_lookback_days: int = MAX_METRIC_LOOKBACK_DAYS,
) -> Metrics:
    """Calculate latest metrics from top level metrics data.

    Args:
        data: Top level metrics timeseries data.
        test_positivity_method: Optional details about how test positivity was calculated.
        max_lookback_days: Number of days back from the latest day to consider metrics.

    Returns: Metrics
    """
    data = data.set_index(CommonFields.DATE)
    metrics = {
        "testPositivityRatioDetails": test_positivity_method,
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
