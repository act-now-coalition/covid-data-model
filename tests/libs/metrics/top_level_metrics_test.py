from typing import List
import dataclasses
from typing import Optional

import numpy as np
import pandas as pd
import structlog
from covidactnow.datapublic.common_fields import CommonFields

import libs.metrics.test_positivity
from api import can_api_v2_definition
from libs.metrics import top_level_metrics
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.pipeline import Region

from tests.dataset_utils_test import read_csv_and_index_fips_date
from tests.test_helpers import DEFAULT_REGION
from tests.test_helpers import build_one_region_dataset


# Columns used for building input dataframes in tests. It covers the fields
# required to run top level metrics without error.
INPUT_COLUMNS = [
    CommonFields.NEW_CASES,
    CommonFields.CASES,
    CommonFields.TEST_POSITIVITY,
    CommonFields.POSITIVE_TESTS,
    CommonFields.NEGATIVE_TESTS,
    CommonFields.CONTACT_TRACERS_COUNT,
    CommonFields.CURRENT_ICU,
    CommonFields.ICU_BEDS,
    CommonFields.CURRENT_ICU_TOTAL,
    CommonFields.VACCINATIONS_INITIATED,
    CommonFields.VACCINATIONS_COMPLETED,
]


def build_metrics_df(
    fips: str,
    *,
    start_date: Optional[str] = None,
    dates: Optional[List[str]] = None,
    **metrics_data,
) -> pd.DataFrame:
    """Builds a dataframe that has same structure as those built by
    `top_level_metrics.calculate_metrics_for_timeseries`

    Check out test_helpers.build_dataset if you want a MultiRegionDataset.

    Args:
        fips: Fips code for region.
        start_date: Optional start date.
        dates: Optional list of dates to use for rows, length of dates must match
            length of data in column_data.
        column_data: Column data with values.  Names of variables should match columns
            in `all_columns`. All lists must be the same length.
    """
    metrics = [metric for metric in top_level_metrics.MetricsFields]
    data = {column: metrics_data.get(column, np.nan) for column in metrics}

    max_len = max(len(value) for value in data.values() if isinstance(value, list))

    if dates:
        dates = [pd.Timestamp(date) for date in dates]
    elif start_date:
        dates = pd.date_range(start_date, periods=max_len, freq="D")
    else:
        raise ValueError("Must set either `start_date` or `dates`")

    return pd.DataFrame({"date": dates, "fips": fips, **data})


def _series_with_date_index(data, date: str = "2020-08-25"):
    date_series = pd.date_range(date, periods=len(data), freq="D", name=CommonFields.DATE)
    return pd.Series(data, index=date_series, dtype="float")


def _fips_csv_to_one_region(
    csv_str: str, region: Region, latest=None
) -> OneRegionTimeseriesDataset:
    df = read_csv_and_index_fips_date(csv_str).reset_index()
    # from_timeseries_and_latest adds the location_id column needed by get_one_region
    dataset = MultiRegionDataset.from_fips_timeseries_df(df).get_one_region(region)
    if latest:
        return dataclasses.replace(dataset, latest=latest)
    else:
        return dataset


def test_calculate_case_density():
    """
    It should use population, smoothing and a normalizing factor to calculate case density.
    """
    cases = _series_with_date_index([56, 68, 68, 11, 37, 32, 73, 103, 109, 105, None, 182, 238])
    pop = 100000
    every_ten = 100000
    smooth = 7

    density = top_level_metrics.calculate_case_density(
        cases, pop, smooth=smooth, normalize_by=every_ten
    )
    return density
    pd.testing.assert_series_equal(
        density, _series_with_date_index([0.0, 0.0, 1.0, 3.0, 5.0]),
    )


def test_calculate_test_positivity():
    """
    It should use smoothed case data to calculate positive test rate.
    """
    both = build_one_region_dataset(
        {CommonFields.POSITIVE_TESTS: [0, 1, 3, 6], CommonFields.NEGATIVE_TESTS: [0, 0, 1, 3]}
    )
    positive_rate = libs.metrics.test_positivity.calculate_test_positivity(both, lag_lookback=1)
    expected = _series_with_date_index([np.nan, 1, 0.75, 2 / 3])
    pd.testing.assert_series_equal(positive_rate, expected)


def test_calculate_test_positivity_lagging():
    """
    It should return an empty series if there is missing negative case data.
    """
    both = build_one_region_dataset(
        {CommonFields.POSITIVE_TESTS: [0, 1, 2, 4, 8], CommonFields.NEGATIVE_TESTS: [0, 0, 1, 2, 2]}
    )
    positive_rate = libs.metrics.test_positivity.calculate_test_positivity(both, lag_lookback=1)
    pd.testing.assert_series_equal(positive_rate, pd.Series([], dtype="float64"))


def test_calculate_test_positivity_extra_day():
    """
    It should return an empty series if there is missing negative case data.
    """
    both = build_one_region_dataset(
        {CommonFields.POSITIVE_TESTS: [0, 4, np.nan], CommonFields.NEGATIVE_TESTS: [0, 4, np.nan]}
    )
    positive_rate = libs.metrics.test_positivity.calculate_test_positivity(both)
    expected = _series_with_date_index([np.nan, 0.5])
    pd.testing.assert_series_equal(positive_rate, expected)


def test_top_level_metrics_basic():
    metrics = {
        CommonFields.CASES: [10, 20, None, 40],
        CommonFields.NEW_CASES: [10, 10, None, None],
        CommonFields.TEST_POSITIVITY: [None, 0.1, 0.1, 0.1],
        CommonFields.CONTACT_TRACERS_COUNT: [1, 2, 3, 4],
        CommonFields.CURRENT_ICU: [10, 10, 10, 10],
        CommonFields.CURRENT_ICU_TOTAL: [20, 20, 20, 20],
        CommonFields.ICU_BEDS: [None, None, None, None],
        CommonFields.VACCINATIONS_INITIATED_PCT: [1, 2, None, 3],
        CommonFields.VACCINATIONS_COMPLETED_PCT: [0.1, 0.2, None, 0.3],
    }
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.STATE: "NY",
        CommonFields.ICU_BEDS: 30,
    }
    one_region = build_one_region_dataset(
        metrics, start_date="2020-08-17", timeseries_columns=INPUT_COLUMNS, latest_override=latest,
    )
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        one_region, None, structlog.get_logger()
    )

    expected = build_metrics_df(
        DEFAULT_REGION.fips,
        start_date="2020-08-17",
        caseDensity=[10, 10, None, None],
        testPositivityRatio=[None, 0.1, 0.1, 0.1],
        contactTracerCapacityRatio=[0.02, 0.04, None, None],
        vaccinationsInitiatedRatio=[0.01, 0.02, None, 0.03],
        vaccinationsCompletedRatio=[0.001, 0.002, None, 0.003],
    )
    pd.testing.assert_frame_equal(expected, results)


def test_top_level_metrics_rounding():
    metrics = {
        CommonFields.CASES: [1, 2, 5],
        CommonFields.NEW_CASES: [1, 1, 3],
        CommonFields.TEST_POSITIVITY: [0.1, 0.10051, 0.10049],
        CommonFields.CONTACT_TRACERS_COUNT: [1, 2, 3],
        CommonFields.CURRENT_ICU: [10, 20, 30],
        CommonFields.CURRENT_ICU_TOTAL: [10, 20, 30],
        CommonFields.ICU_BEDS: [None, None, None],
        CommonFields.VACCINATIONS_INITIATED_PCT: [33.3333, 66.666, 100],
        CommonFields.VACCINATIONS_COMPLETED_PCT: [33.3333, 66.6666, 100],
    }
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.STATE: "NY",
        CommonFields.ICU_BEDS: 30,
    }
    one_region = build_one_region_dataset(
        metrics, start_date="2020-08-17", timeseries_columns=INPUT_COLUMNS, latest_override=latest,
    )
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        one_region, None, structlog.get_logger()
    )

    expected = build_metrics_df(
        DEFAULT_REGION.fips,
        start_date="2020-08-17",
        caseDensity=[1, 1, 1.7],
        testPositivityRatio=[0.1, 0.101, 0.1],
        contactTracerCapacityRatio=[0.2, 0.4, 0.36],
        vaccinationsInitiatedRatio=[0.333, 0.667, 1],
        vaccinationsCompletedRatio=[0.333, 0.667, 1],
    )
    pd.testing.assert_frame_equal(expected, results)


def test_top_level_metrics_incomplete_latest():
    region_ny = Region.from_state("NY")
    # This test doesn't have ICU_BEDS set in `latest`. It checks that the metrics are still built.
    metrics = {
        CommonFields.CASES: [10, 20, None, 40],
        CommonFields.NEW_CASES: [10, 10, 10, 10],
        CommonFields.TEST_POSITIVITY: [None, 0.1, 0.1, 0.1],
        CommonFields.CONTACT_TRACERS_COUNT: [1, 2, 3, 4],
        CommonFields.CURRENT_ICU: [10, 10, 10, 10],
        CommonFields.CURRENT_ICU_TOTAL: [20, 20, 20, 20],
        CommonFields.ICU_BEDS: [None, None, None, None],
    }
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.STATE: "NY",
        # ICU_BEDS not set
    }
    one_region = build_one_region_dataset(
        metrics,
        region=region_ny,
        start_date="2020-08-17",
        timeseries_columns=INPUT_COLUMNS,
        latest_override=latest,
    )
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        one_region, None, structlog.get_logger()
    )

    expected = build_metrics_df(
        "36",
        start_date="2020-08-17",
        caseDensity=[10, 10, 10, 10],
        testPositivityRatio=[None, 0.1, 0.1, 0.1],
        contactTracerCapacityRatio=[0.02, 0.04, 0.06, 0.08],
    )
    pd.testing.assert_frame_equal(expected, results, check_dtype=False)


def test_top_level_metrics_no_pos_neg_tests_no_positivity_ratio():
    region_ny = Region.from_state("NY")
    # All of positive_tests, negative_tests are empty and test_positivity is absent. Make sure
    # other metrics are still produced.
    metrics = {
        CommonFields.CASES: [10, 20, 30, 40],
        CommonFields.NEW_CASES: [10, 10, 10, 10],
        CommonFields.CONTACT_TRACERS_COUNT: [1, 2, 3, 4],
    }
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
        CommonFields.STATE: "NY",
        CommonFields.ICU_BEDS: 10,
    }
    one_region = build_one_region_dataset(
        metrics,
        region=region_ny,
        start_date="2020-08-17",
        timeseries_columns=INPUT_COLUMNS,
        latest_override=latest,
    )
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        one_region, None, structlog.get_logger()
    )

    expected = build_metrics_df(
        "36",
        start_date="2020-08-17",
        caseDensity=[10.0, 10.0, 10.0, 10.0],
        contactTracerCapacityRatio=[0.02, 0.04, 0.06, 0.08],
    )
    pd.testing.assert_frame_equal(expected, results)


def test_top_level_metrics_no_pos_neg_tests_has_positivity_ratio():
    ny_region = Region.from_state("NY")
    metrics = {
        CommonFields.CASES: [10, 20, 30, 40],
        CommonFields.NEW_CASES: [10, 10, 10, 10],
        CommonFields.TEST_POSITIVITY: [0.02, 0.03, 0.04, 0.05],
    }
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
        CommonFields.STATE: "NY",
        CommonFields.ICU_BEDS: 10,
    }

    one_region = build_one_region_dataset(
        metrics,
        start_date="2020-08-17",
        timeseries_columns=INPUT_COLUMNS,
        latest_override=latest,
        region=ny_region,
    )

    # All of positive_tests, negative_tests are empty. test_positivity has a real value. Make sure
    # test_positivity is copied to the output and other metrics are produced.
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        one_region, None, structlog.get_logger()
    )

    expected = build_metrics_df(
        "36",
        start_date="2020-08-17",
        caseDensity=[10, 10, 10, 10],
        testPositivityRatio=[0.02, 0.03, 0.04, 0.05],
    )
    pd.testing.assert_frame_equal(expected, results, check_dtype=False)


def test_top_level_metrics_with_rt():
    region = Region.from_fips("36")
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
        CommonFields.STATE: "NY",
        CommonFields.ICU_BEDS: 25,
    }

    metrics = {
        CommonFields.NEW_CASES: [None, 10, None, None],
        CommonFields.TEST_POSITIVITY: [None, 0.1, 0.1, 0.1],
        CommonFields.CONTACT_TRACERS_COUNT: [1, 2, 3, 4],
    }

    one_region = build_one_region_dataset(
        metrics,
        start_date="2020-08-17",
        timeseries_columns=INPUT_COLUMNS,
        latest_override=latest,
        region=region,
    )
    data = (
        "date,fips,Rt_MAP_composite,Rt_ci95_composite\n"
        "2020-08-17,36,1.1,1.2\n"
        "2020-08-18,36,1.2,1.3\n"
        "2020-08-19,36,1.1,1.3\n"
        "2020-08-20,36,1.1,1.2\n"
    )
    rt_data = _fips_csv_to_one_region(data, region)

    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        one_region, rt_data, structlog.get_logger()
    )
    expected = build_metrics_df(
        "36",
        start_date="2020-08-17",
        caseDensity=[0, 5, None, None],
        testPositivityRatio=[None, 0.1, 0.1, 0.1],
        contactTracerCapacityRatio=[None, 0.08, None, None],
        infectionRate=[1.1, 1.2, 1.1, 1.1],
        infectionRateCI90=[0.1, 0.1, 0.2, 0.1],
    )
    pd.testing.assert_frame_equal(expected, results)


def test_calculate_contact_tracers():

    cases = _series_with_date_index([0.0, 1.0, 3.0])
    contact_tracers = _series_with_date_index([5, 5, 5])

    results = top_level_metrics.calculate_contact_tracers(cases, contact_tracers)
    expected = _series_with_date_index([np.nan, 2.0, 0.75])
    pd.testing.assert_series_equal(results, expected)


def test_calculate_contact_tracers_no_tracers():

    cases = _series_with_date_index([0.0, 1.0, 4.0])
    contact_tracers = _series_with_date_index([np.nan, np.nan, np.nan])

    results = top_level_metrics.calculate_contact_tracers(cases, contact_tracers)
    expected = _series_with_date_index([np.nan, np.nan, np.nan])
    pd.testing.assert_series_equal(results, expected)


def test_calculate_latest_rt():
    prev_rt = 1.0
    prev_rt_ci90 = 0.2
    data = build_metrics_df(
        "36",
        dates=["2020-08-13", "2020-08-20"],
        caseDensity=[10, 10],
        testPositivityRatio=[0.1, 0.1],
        contactTracerCapacityRatio=[0.06, 0.08],
        infectionRate=[prev_rt, 2.0],
        infectionRateCI90=[prev_rt_ci90, 0.2],
    )
    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert metrics.infectionRate == prev_rt
    assert metrics.infectionRateCI90 == prev_rt_ci90


def test_lookback_days():
    prev_rt = 1.0
    prev_rt_ci90 = 0.2
    data = build_metrics_df(
        "36",
        dates=["2020-08-12", "2020-08-13", "2020-08-28"],
        caseDensity=[10, 10, None],
        testPositivityRatio=[0.1, 0.1, None],
        contactTracerCapacityRatio=[0.06, 0.6, None],
        infectionRate=[prev_rt, None, None],
        infectionRateCI90=[prev_rt_ci90, None, None],
    )

    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert not metrics.caseDensity
    assert not metrics.testPositivityRatio
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90


def test_calculate_latest_rt_no_previous_row():
    data = build_metrics_df(
        "36",
        start_date="2020-08-20",
        caseDensity=[10],
        testPositivityRatio=[0.1],
        contactTracerCapacityRatio=[0.08],
        infectionRate=[2.0],
        infectionRateCI90=[0.2],
    )
    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90


def test_calculate_latest_rt_no_rt():
    data = build_metrics_df(
        "36",
        start_date="2020-08-20",
        caseDensity=[10],
        testPositivityRatio=[0.1],
        contactTracerCapacityRatio=[0.08],
    )
    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90


def test_calculate_latest_different_latest_days():
    prev_rt = 1.0
    prev_rt_ci90 = 0.2
    data = build_metrics_df(
        "36",
        dates=["2020-08-13", "2020-08-20"],
        caseDensity=[10, None],
        testPositivityRatio=[0.1, 0.2],
        contactTracerCapacityRatio=[0.06, 0.08],
        infectionRate=[prev_rt, 2.01],
        infectionRateCI90=[prev_rt_ci90, 0.2],
        icuCapacityRatio=0.75,
    )
    expected_metrics = can_api_v2_definition.Metrics(
        testPositivityRatio=0.2,
        caseDensity=10,
        contactTracerCapacityRatio=0.08,
        infectionRate=prev_rt,
        infectionRateCI90=prev_rt_ci90,
        icuCapacityRatio=0.75,
    )
    metrics = top_level_metrics.calculate_latest_metrics(data, None, max_lookback_days=8)
    assert metrics == expected_metrics


def test_calculate_icu_capacity():
    region = Region.from_fips("36")
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
        CommonFields.STATE: "NY",
    }
    one_region = build_one_region_dataset(
        {CommonFields.ICU_BEDS: [10, 20], CommonFields.CURRENT_ICU_TOTAL: [10, 15]},
        region=region,
        start_date="2020-12-18",
        timeseries_columns=INPUT_COLUMNS,
        latest_override=latest,
    )

    results, metrics = top_level_metrics.calculate_metrics_for_timeseries(
        one_region, None, structlog.get_logger()
    )
    expected = build_metrics_df("36", start_date="2020-12-18", icuCapacityRatio=[1.0, 0.75],)
    positivity_method = can_api_v2_definition.TestPositivityRatioDetails(
        source=can_api_v2_definition.TestPositivityRatioMethod.OTHER
    )
    expected_metrics = top_level_metrics.calculate_latest_metrics(expected, positivity_method)
    pd.testing.assert_frame_equal(expected, results)
    assert metrics == expected_metrics
