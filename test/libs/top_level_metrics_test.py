import unittest

import numpy as np
import pandas as pd

from libs.top_level_metrics import (
    calculate_case_density,
    calculate_test_positivity,
    smooth_with_rolling_average,
)


def test_calculate_case_density():
    """
    It should use population, smoothing and a normalizing factor to calculate case density.
    """
    cases = pd.Series([0, 20, 40, 60])
    pop = 100
    every_ten = 10
    smooth = 2

    density = calculate_case_density(
        cases=cases, population=pop, smooth=smooth, normalize_by=every_ten
    )
    assert density.equals(pd.Series([0, 1, 2, 3], dtype="float"))


def test_calculate_test_positivity():
    """
    It should use smoothed case data to calculate positive test rate.
    """

    pos_cases = pd.Series([1, 2, 3])
    neg_cases = pd.Series([0, 1, 2])
    pos_rate = calculate_test_positivity(
        pos_cases=pos_cases, neg_tests=neg_cases, smooth=2, lag_lookback=1
    )
    assert pos_rate.equals(pd.Series([1, 0.75, 2 / 3], dtype="float64"))


def test_calculate_test_positivity_lagging():
    """
    It should return an empty series if there is missing negative case data.
    """
    pos_cases = pd.Series([1, 2, 4, 8])
    neg_cases = pd.Series([0, 1, 2, np.nan])
    pos_rate = calculate_test_positivity(
        pos_cases=pos_cases, neg_tests=neg_cases, smooth=2, lag_lookback=1
    )
    assert pos_rate.equals(pd.Series([], dtype="float64"))


def test_sliding_window():
    """
    It should average within sliding window.
    """
    series = pd.Series([1, 2, 4, 5, 0])
    smoothed = smooth_with_rolling_average(series=series, window=2)
    assert smoothed.equals, pd.Series([1, 1.5, 3, 4.5, 2.5])


def test_window_exceeds_series():
    """
    It should not average the first value.
    """
    series = pd.Series([1, 2])
    smoothed = smooth_with_rolling_average(series=series, window=5)
    assert smoothed.equals, pd.Series([1, 1.5])


def test_exclude_trailing_zeroes():
    """
    It should not average the first value.
    """
    series = pd.Series([1, 2, 4, 5, 0])
    smoothed = smooth_with_rolling_average(series=series, window=5, includeTrailingZeros=False)
    assert smoothed.equals, pd.Series([1, 1.5, 3, 4.5, np.nan])
