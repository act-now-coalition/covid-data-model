import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
from libs import series_utils


def _series_with_date_index(data, date: str = "2020-08-25", **series_kwargs):
    date_series = pd.date_range(date, periods=len(data), freq="D")
    return pd.Series(data, index=date_series, **series_kwargs)


def test_sliding_window():
    """
    It should average within sliding window.
    """
    series = _series_with_date_index([1, 2, 4, 5, 0])
    smoothed = series_utils.smooth_with_rolling_average(series=series, window=2)
    pd.testing.assert_series_equal(smoothed, _series_with_date_index([1, 1.5, 3, 4.5, 2.5]))


def test_window_exceeds_series():
    """
    It should not average the first value.
    """
    series = _series_with_date_index([1, 2])
    smoothed = series_utils.smooth_with_rolling_average(series=series, window=5)
    pd.testing.assert_series_equal(smoothed, _series_with_date_index([1, 1.5]))


def test_exclude_trailing_zeroes():
    """
    It should not average the first value.
    """
    series = _series_with_date_index([1, 2, 4, 5, 0])
    smoothed = series_utils.smooth_with_rolling_average(
        series=series, window=2, include_trailing_zeros=False
    )
    pd.testing.assert_series_equal(smoothed, _series_with_date_index([1, 1.5, 3, 4.5, np.nan]))


def test_interpolation():

    series = _series_with_date_index([np.nan, 1, 2, 2, 4])

    results = series_utils.interpolate_stalled_values(series)
    expected = _series_with_date_index([np.nan, 1, 2, 3, 4])
    pd.testing.assert_series_equal(results, expected)


def test_interpolation_bounds():

    series = _series_with_date_index([np.nan, 1, 2, 2, 4, np.nan])

    results = series_utils.interpolate_stalled_values(series)
    expected = _series_with_date_index([np.nan, 1, 2, 3, 4, np.nan])
    pd.testing.assert_series_equal(results, expected)


def test_interpolation_missing_index():
    index = pd.date_range("2020-08-25", periods=6).drop([pd.Timestamp("2020-08-29")])
    series = pd.Series([np.nan, 1, 2, 2, 5], index=index)

    results = series_utils.interpolate_stalled_values(series)
    expected = pd.Series([np.nan, 1, 2, 3, 5], index=index)
    pd.testing.assert_series_equal(results, expected)


def test_has_recent_data():
    start_date = datetime.datetime.today().date() - timedelta(days=15)
    series = _series_with_date_index([0] * 15, date=start_date)
    assert not series_utils.has_recent_data(series)

    series = _series_with_date_index([1] * 15, date=start_date)
    assert series_utils.has_recent_data(series)

    series = _series_with_date_index([1] * 7, date=start_date)
    assert not series_utils.has_recent_data(series)

    series = _series_with_date_index([1] * 8, date=start_date)
    assert series_utils.has_recent_data(series)

    series = _series_with_date_index([np.nan] * 8, date=start_date)
    assert not series_utils.has_recent_data(series)

    series = _series_with_date_index([1] * 8, date=start_date)
    assert not series_utils.has_recent_data(series, days_back=5)
