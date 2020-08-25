import io

import pytest
import numpy as np
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
from libs import top_level_metrics
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput


def _build_metrics_df(content: str) -> pd.DataFrame:
    header = (
        "date,fips,caseDensity,testPositivityRatio,contactTracerCapacityRatio,"
        "infectionRate,infectionRateCI90\n"
    )
    data = io.StringIO(f"{header}\n{content}")
    return common_df.read_csv(data, set_index=False)


def test_calculate_case_density():
    """
    It should use population, smoothing and a normalizing factor to calculate case density.
    """
    cases = pd.Series([0, 0, 20, 60, 120])
    pop = 100
    every_ten = 10
    smooth = 2

    density = top_level_metrics.calculate_case_density(
        cases, pop, smooth=smooth, normalize_by=every_ten
    )
    assert density.equals(pd.Series([np.nan, 0, 1, 2, 3], dtype="float"))


def test_calculate_test_positivity():
    """
    It should use smoothed case data to calculate positive test rate.
    """

    positive_tests = pd.Series([0, 1, 3, 6])
    negative_tests = pd.Series([0, 0, 1, 3])
    positive_rate = top_level_metrics.calculate_test_positivity(
        positive_tests, negative_tests, smooth=2, lag_lookback=1
    )
    expected = pd.Series([np.nan, 1, 0.75, 2 / 3], dtype="float64")
    pd.testing.assert_series_equal(positive_rate, expected)


def test_calculate_test_positivity_lagging():
    """
    It should return an empty series if there is missing negative case data.
    """
    positive_tests = pd.Series([0, 1, 2, 4, 8])
    negative_tests = pd.Series([0, 0, 1, 2, 2])
    positive_rate = top_level_metrics.calculate_test_positivity(
        positive_tests, negative_tests, smooth=2, lag_lookback=1
    )
    pd.testing.assert_series_equal(positive_rate, pd.Series([], dtype="float64"))


def test_calculate_test_positivity_extra_day():
    """
    It should return an empty series if there is missing negative case data.
    """
    positive_tests = pd.Series([0, 4, np.nan])
    negative_tests = pd.Series([0, 4, np.nan])
    positive_rate = top_level_metrics.calculate_test_positivity(positive_tests, negative_tests)
    pd.testing.assert_series_equal(positive_rate, pd.Series([np.nan, 0.5], dtype="float64"))


def test_top_level_metrics_basic():
    data = io.StringIO(
        "date,fips,cases,positive_tests,negative_tests,contact_tracers_count\n"
        "2020-08-17,36,10,10,90,1\n"
        "2020-08-18,36,20,20,180,2\n"
        "2020-08-19,36,,,,3\n"
        "2020-08-20,36,40,40,360,4\n"
    )
    timeseries = TimeseriesDataset.load_csv(data)
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
    }
    results = top_level_metrics.calculate_metrics_for_timeseries(timeseries, latest, None)

    header = (
        "date,fips,caseDensity,testPositivityRatio,contactTracerCapacityRatio,"
        "infectionRate,infectionRateCI90\n"
    )
    expected = io.StringIO(
        f"{header}"
        "2020-08-17,36,,,,,\n"
        "2020-08-18,36,10,0.1,0.04,,\n"
        "2020-08-19,36,10,0.1,0.06,,\n"
        "2020-08-20,36,10,0.1,0.08,,\n"
    )
    expected = TimeseriesDataset.load_csv(expected).data
    pd.testing.assert_frame_equal(expected, results)


def test_top_level_metrics_with_rt():
    data = io.StringIO(
        "date,fips,cases,positive_tests,negative_tests,contact_tracers_count\n"
        "2020-08-17,36,10,10,90,1\n"
        "2020-08-18,36,20,20,180,2\n"
        "2020-08-19,36,,,,3\n"
        "2020-08-20,36,40,40,360,4\n"
    )
    timeseries = TimeseriesDataset.load_csv(data)

    data = io.StringIO(
        "date,fips,Rt_indicator,Rt_indicator_ci90,intervention,all_hospitalized,beds\n"
        "2020-08-17,36,1.1,.1,0,1,10\n"
        "2020-08-18,36,1.2,.1,0,1,10\n"
        "2020-08-19,36,1.1,.2,0,1,10\n"
        "2020-08-20,36,1.1,.1,0,1,10\n"
        "2020-09-21,36,,,0,1,10\n"
    )
    data = common_df.read_csv(data, set_index=False)
    model_output = CANPyseirLocationOutput(data)

    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
    }
    results = top_level_metrics.calculate_metrics_for_timeseries(timeseries, latest, model_output)
    header = (
        "date,fips,caseDensity,testPositivityRatio,contactTracerCapacityRatio,"
        "infectionRate,infectionRateCI90\n"
    )
    expected = io.StringIO(
        f"{header}"
        "2020-08-17,36,,,,1.1,.1\n"
        "2020-08-18,36,10,0.1,0.04,1.2,.1\n"
        "2020-08-19,36,10,0.1,0.06,1.1,.2\n"
        "2020-08-20,36,10,0.1,0.08,1.1,.1\n"
    )
    expected = TimeseriesDataset.load_csv(expected).data
    pd.testing.assert_frame_equal(expected, results)


def test_calculate_contact_tracers():

    cases = pd.Series([0.0, 1.0, 4.0])
    contact_tracers = pd.Series([5, 5, 5])

    results = top_level_metrics.calculate_contact_tracers(cases, contact_tracers)
    expected = pd.Series([np.nan, 1.0, 0.5])
    pd.testing.assert_series_equal(results, expected)


def test_calculate_contact_tracers_no_tracers():

    cases = pd.Series([0.0, 1.0, 4.0])
    contact_tracers = pd.Series([np.nan, np.nan, np.nan])

    results = top_level_metrics.calculate_contact_tracers(cases, contact_tracers)
    expected = pd.Series([np.nan, np.nan, np.nan])
    pd.testing.assert_series_equal(results, expected)


def test_calculate_latest_rt():
    prev_rt = 1.0
    prev_rt_ci90 = 0.2
    data = _build_metrics_df(
        f"2020-08-13,36,10,0.1,0.06,{prev_rt},{prev_rt_ci90}\n"
        "2020-08-20,36,10,0.1,0.08,2.0,0.2\n"
    )
    metrics = top_level_metrics.calculate_latest_metrics(data)
    assert metrics.infectionRate == prev_rt
    assert metrics.infectionRateCI90 == prev_rt_ci90


def test_calculate_latest_rt_no_previous_row():
    data = _build_metrics_df("2020-08-20,36,10,0.1,0.08,2.0,0.2\n")
    metrics = top_level_metrics.calculate_latest_metrics(data)
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90


def test_calculate_latest_rt_no_rt():
    data = _build_metrics_df("2020-08-20,36,10,0.1,0.08,,\n")
    metrics = top_level_metrics.calculate_latest_metrics(data)
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90
