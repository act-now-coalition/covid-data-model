from datetime import timedelta
import datetime
import pandas as pd
import numpy as np


def smooth_with_rolling_average(
    series: pd.Series,
    window: int = 7,
    include_trailing_zeros: bool = True,
    exclude_negatives: bool = True,
):
    """Smoothes series with a min period of 1.

    Series must have a datetime index.

    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rolling.html

    Port of Projections.ts:
    https://github.com/act-now-coalition/covid-projections/blob/master/src/common/models/Projection.ts#L715

    Args:
        series: Series with datetime index to smooth.
        window: Sliding window to average.
        include_trailing_zeros: Whether or not to NaN out trailing zeroes.
        exclude_negatives: Exclude negative values from rolling averages.

    Returns:
        Smoothed series.
    """
    # Drop trailing NAs so that we don't smooth for day we don't yet have data.
    series = series.loc[: series.last_valid_index()]

    if exclude_negatives:
        series = series.copy()
        series.loc[series < 0] = None

    def mean_with_no_trailing_nan(x):
        """Return mean of series unless last value is nan."""
        if np.isnan(x.iloc[-1]):
            return np.nan

        return x.mean()

    # Apply function to a rolling window
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.window.rolling.Rolling.apply.html
    rolling_average = series.rolling(window, min_periods=1).apply(mean_with_no_trailing_nan)
    if include_trailing_zeros:
        return rolling_average

    last_valid_index = series.replace(0, np.nan).last_valid_index()

    if last_valid_index:
        rolling_average[last_valid_index + timedelta(days=1) :] = np.nan
        return rolling_average
    else:  # entirely empty series:
        return series


def interpolate_stalled_and_missing_values(series: pd.Series) -> pd.Series:
    """Interpolates periods where values have stopped increasing or have gaps.

    Args:
        series: Series with a datetime index
    """
    series = series.copy()
    start, end = series.first_valid_index(), series.last_valid_index()
    series_with_values = series.loc[start:end]

    series_with_values[series_with_values.diff() == 0] = None
    # Use the index to determine breaks between data (so
    # missing data is not improperly interpolated)
    series.loc[start:end] = series_with_values.interpolate(method="time").apply(np.floor)

    return series


def has_recent_data(
    series: pd.Series, days_back: int = 14, required_non_null_datapoints: int = 7
) -> bool:
    """Checks to see if series has recent non-null data with at least one non-zero data point

    Args:
        series: Series with a datetime index.
        days_back: Number of days back to look
        required_non_null_datapoints: Number of non-null data points required.

    Returns: True if has recent data, otherwise false.
    """
    today = datetime.datetime.today().date()
    start_date = today - timedelta(days=days_back)
    recent_series = series[start_date:]
    num_datapoints = recent_series.notnull().sum()
    has_nonzero_points = (recent_series > 0).any()
    return num_datapoints >= required_non_null_datapoints and has_nonzero_points
