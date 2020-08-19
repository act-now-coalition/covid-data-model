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
