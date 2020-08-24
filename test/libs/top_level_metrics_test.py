import io
import pytest
import numpy as np
import pandas as pd
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs import top_level_metrics
from libs.datasets.timeseries import TimeseriesDataset


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
    results = top_level_metrics.calculate_top_level_metrics_for_timeseries(timeseries, latest)
    expected = io.StringIO(
        "date,fips,caseDensity,testPositivityRatio,contactTracerCapacityRatio\n"
        "2020-08-17,36,,,\n"
        "2020-08-18,36,10,0.1,0.04\n"
        "2020-08-19,36,10,0.1,0.06\n"
        "2020-08-20,36,10,0.1,0.08\n"
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
