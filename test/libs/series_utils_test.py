import pandas as pd
import numpy as np
from libs import series_utils


def test_sliding_window():
    """
    It should average within sliding window.
    """
    series = pd.Series([1, 2, 4, 5, 0])
    smoothed = series_utils.smooth_with_rolling_average(series=series, window=2)
    pd.testing.assert_series_equal(smoothed, pd.Series([1, 1.5, 3, 4.5, 2.5]))


def test_window_exceeds_series():
    """
    It should not average the first value.
    """
    series = pd.Series([1, 2])
    smoothed = series_utils.smooth_with_rolling_average(series=series, window=5)
    pd.testing.assert_series_equal(smoothed, pd.Series([1, 1.5]))


def test_exclude_trailing_zeroes():
    """
    It should not average the first value.
    """
    series = pd.Series([1, 2, 4, 5, 0])
    smoothed = series_utils.smooth_with_rolling_average(
        series=series, window=2, include_trailing_zeros=False
    )
    pd.testing.assert_series_equal(smoothed, pd.Series([1, 1.5, 3, 4.5, np.nan]))


def test_interpolation():

    series = pd.Series([np.nan, 1, 2, 2, 4])

    results = series_utils.interpolate_stalled_values(series)
    expected = pd.Series([np.nan, 1, 2, 3, 4])
    pd.testing.assert_series_equal(results, expected)


def test_interpolation_bounds():

    series = pd.Series([np.nan, 1, 2, 2, 4, np.nan])

    results = series_utils.interpolate_stalled_values(series)
    expected = pd.Series([np.nan, 1, 2, 3, 4, np.nan])
    pd.testing.assert_series_equal(results, expected)


def test_interpolation_missing_index():

    series = pd.Series([np.nan, 1, 2, 2, 5], index=[0, 1, 2, 3, 5])

    results = series_utils.interpolate_stalled_values(series)
    expected = pd.Series([np.nan, 1, 2, 3, 5], index=[0, 1, 2, 3, 5])
    pd.testing.assert_series_equal(results, expected)
