import unittest

import numpy as np
import pandas as pd

from libs import top_level_metrics


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
    negative_tests = pd.Series([0, 0, 1, 2, np.nan])
    positive_rate = top_level_metrics.calculate_test_positivity(
        positive_tests, negative_tests, smooth=2, lag_lookback=1
    )
    pd.testing.assert_series_equal(positive_rate, pd.Series([], dtype="float64"))
