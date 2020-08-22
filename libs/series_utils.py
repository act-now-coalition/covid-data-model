from datetime import timedelta
import pandas as pd
import numpy as np


def smooth_with_rolling_average(
    series: pd.Series,
    window: int = 7,
    include_trailing_zeros: bool = True,
    exclude_negatives: bool = True,
):
    """Smoothes series with a min period of 1.

    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rolling.html

    Port of Projections.ts:
    https://github.com/covid-projections/covid-projections/blob/master/src/common/models/Projection.ts#L715

    Args:
        series: Series to smooth.
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

    rolling_average = series.rolling(window, min_periods=1).mean()
    if include_trailing_zeros:
        return rolling_average

    last_valid_index = series.replace(0, np.nan).last_valid_index()

    if last_valid_index:
        rolling_average[last_valid_index + 1 :] = np.nan
        return rolling_average
    else:  # entirely empty series:
        return series


def interpolate_stalled_values(series: pd.Series) -> pd.Series:
    """Interpolates periods where values have stopped increasing,

    Args:
        series: Series

    """
    series = series.copy()
    start, end = series.first_valid_index(), series.last_valid_index()
    series_with_values = series.loc[start:end]

    series_with_values[(series_with_values.diff() == 0)] = None
    # Use the index to determine breaks between data (so
    # missing data is not improperly interpolated)
    series.loc[start:end] = series_with_values.interpolate(method="index").round()

    return series
