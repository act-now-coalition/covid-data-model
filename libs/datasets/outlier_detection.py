from typing import Optional, Tuple
import dataclasses

import pandas as pd
import numpy as np
from datapublic.common_fields import CommonFields
from datapublic.common_fields import PdFields

from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets import new_cases_and_deaths
from libs.datasets.dataset_utils import AggregationLevel
from libs.pipeline import RegionMask

MultiRegionDataset = timeseries.MultiRegionDataset

# Florida biweekly case reporting causes outlier detection to trigger inadvertently.
STATES_TO_EXCLUDE_CASES_DEATHS = [RegionMask(AggregationLevel.COUNTY, states=["FL"])]


def _calculate_modified_zscore(
    series: pd.Series,
    window: int = 10,
    min_periods=3,
    ignore_zeros=True,
    spread_first_report_after_zeros=True,
) -> pd.Series:
    """Calculates zscore for each point in series comparing current point to past `window` days.

    Each datapoint is compared to the distribution of the past `window` days as long as there are
    `min_periods` number of non-nan values in the window.

    In the calculation of z-score, zeros are thrown out. This is done to produce better results
    for regions that regularly report zeros (for instance, RI reports zero new cases on
    each weekend day).

    Args:
        series: Series to compute statistics for.
        window: Size of window to calculate mean and std.
        min_periods: Number of periods necessary to compute a score - will return nan otherwise.
        ignore_zeros: If true, zeros are not included in zscore calculation.

    Returns: Array of scores for each datapoint in series.
    """
    series = series.copy()

    if spread_first_report_after_zeros:
        series = new_cases_and_deaths.spread_first_reported_value_after_stall(series)

    if ignore_zeros:
        series[series == 0] = None

    rolling_series = series.rolling(window=window, min_periods=min_periods)
    # Shifting one to exclude current datapoint
    mean = rolling_series.mean().shift(1)
    std = rolling_series.std(ddof=0).shift(1)
    z = (series - mean) / std
    return z.abs()


def drop_new_case_outliers(timeseries: MultiRegionDataset) -> MultiRegionDataset:
    """Identifies and drops outliers from the new case series.

    Args:
        timeseries: Timeseries.

    Returns: timeseries with outliers removed from new_cases.
    """

    states_to_exclude, dataset_in = timeseries.partition_by_region(STATES_TO_EXCLUDE_CASES_DEATHS)
    if dataset_in.timeseries_bucketed.empty:
        return timeseries  # if dataset only includes locations to ignore, just return the dataset.

    dataset_out = drop_series_outliers(
        dataset_in,
        CommonFields.NEW_CASES,
        zscore_threshold=8.0,
        # As delta ramped up, cases increased at a much quicker rate than before.
        # Because our outlier detection for a given day only looks at the past,
        # new values were rising above the original threshold.  This adjusts the threshold
        # to account for cases rapidly rising as delta became more prevalant.
        secondary_zscore_threshold_after_date=(pd.Timestamp("2021-06-01"), 15.0),
        threshold=30,
    )
    return dataset_out.append_regions(states_to_exclude)


def drop_new_deaths_outliers(timeseries: MultiRegionDataset) -> MultiRegionDataset:
    """Identifies and drops outliers from the new case series.

    Args:
        timeseries: Timeseries.

    Returns: timeseries with outliers removed from new_cases.
    """

    states_to_exclude, dataset_in = timeseries.partition_by_region(STATES_TO_EXCLUDE_CASES_DEATHS)
    if dataset_in.timeseries_bucketed.empty:
        return timeseries  # if dataset only includes locations to ignore, just return the dataset.

    dataset_out = drop_series_outliers(
        dataset_in, CommonFields.NEW_DEATHS, zscore_threshold=8.0, threshold=30,
    )
    return dataset_out.append_regions(states_to_exclude)


def drop_series_outliers(
    dataset: MultiRegionDataset,
    field: CommonFields,
    zscore_threshold: float = 8.0,
    secondary_zscore_threshold_after_date: Optional[Tuple[pd.Timestamp, float]] = None,
    threshold: int = 30,
) -> MultiRegionDataset:
    """Identifies and drops outliers from the new case series.

    Args:
        timeseries: Timeseries.
        zscore_threshold: Z-score threshold.  All new cases with a zscore greater than the
            threshold will be removed.
        secondary_zscore_threshold_after_date: If set, optionally apply a secondary zscore threshold
            after a certain date.
        threshold: Min number of cases needed to count as an outlier.

    Returns: timeseries with outliers removed from new_cases.
    """
    ts_to_filter = dataset.timeseries_bucketed_wide_dates.xs(
        field, level=PdFields.VARIABLE, drop_level=False
    )
    zscores = ts_to_filter.apply(_calculate_modified_zscore, axis=1, result_type="reduce")
    # Around July 4th 2021, lots of places had delayed reporting (no reporting
    # over the weekend or on July 4/5) leading to a valid spike in cases.
    # Simultaneously, the delta variant was starting to take hold, exacerbating
    # the spike and leading to sustained case growth.
    # Because outlier detection removed the first spike in many cases, the
    # resulting r(t) calculation ended up inflated (e.g. >1.5 in San Francisco),
    # so we just disable outlier detection for the days following July 4th.
    dates_to_keep = ["2021-07-05", "2021-07-06", "2021-07-07"]

    if secondary_zscore_threshold_after_date:
        # Create a series with zscore thresholds per date, allowing us to
        # override the zscore for dates after start of secondary zscore threshold.
        zscore_threshold = pd.Series(
            [zscore_threshold] * len(zscores.columns), index=zscores.columns
        )
        secondary_zscore_date, secondary_zscore = secondary_zscore_threshold_after_date
        zscore_threshold[zscore_threshold.index > secondary_zscore_date] = secondary_zscore

    to_exclude = (
        (zscores > zscore_threshold)
        & (ts_to_filter > threshold)
        & ~(ts_to_filter.columns.isin(dates_to_keep))
    )

    return exclude_observations(dataset, to_exclude)


def drop_tail_positivity_outliers(
    dataset: MultiRegionDataset,
    zscore_threshold: float = 10.0,
    diff_threshold_ratio: float = 0.015,
) -> MultiRegionDataset:
    """Drops outliers from the test_positivity_7d series, adding tags for removed values.

    Args:
        dataset:
        zscore_threshold: Z-score threshold.  All test_positivity_7d values with a zscore greater
            than the threshold will be removed.
        diff_threshold_ratio: Minimum difference required for value to be outlier.

    Returns: Dataset with outliers removed from test_positivity_7d.
    """
    # TODO(https://trello.com/c/7J2SmDnr): Be more consistent about accessing this data
    # through wide dates rather than duplicating timeseries.

    if CommonFields.TEST_POSITIVITY_7D not in dataset.timeseries_bucketed_wide_dates.columns:
        return dataset
    ts_to_filter = dataset.timeseries_bucketed_wide_dates.xs(
        CommonFields.TEST_POSITIVITY_7D, level=PdFields.VARIABLE, drop_level=False
    )

    def process_series(series):
        recent_series = series.tail(10)
        # window of 5 days seems to capture about the right amount of variance.
        # If window is too large, there may have been a large enough natural shift
        # in test positivity that recent extreme value looks more noraml.
        series_zscores = _calculate_modified_zscore(recent_series, window=5, ignore_zeros=False)
        series_zscores = series_zscores.dropna().last("1D")
        return series_zscores

    zscores = ts_to_filter.apply(process_series, axis=1, result_type="reduce")
    test_positivity_diffs = ts_to_filter.diff(axis=1).abs()

    to_exclude_wide = (zscores > zscore_threshold) & (test_positivity_diffs > diff_threshold_ratio)
    return exclude_observations(dataset, to_exclude_wide)


def exclude_observations(dataset, to_exclude_wide) -> MultiRegionDataset:
    to_exclude_long = to_exclude_wide.stack()
    to_exclude_index = to_exclude_long.loc[to_exclude_long].index
    new_tags = taglib.TagCollection()
    timeseries_copy = dataset.timeseries_bucketed.copy()
    # Iterate through the MultiIndex to_exclude_index, accessing elements of timeseries_copy.
    # These asserts check that the labels are in the expected order. It might be cleaner to
    # change timeseries_copy to a long format with the same index level order as to_exclude_index
    # but that is a bigger change.
    assert to_exclude_index.names == [
        CommonFields.LOCATION_ID,
        PdFields.VARIABLE,
        PdFields.DEMOGRAPHIC_BUCKET,
        CommonFields.DATE,
    ]
    assert timeseries_copy.index.names == [
        CommonFields.LOCATION_ID,
        PdFields.DEMOGRAPHIC_BUCKET,
        CommonFields.DATE,
    ]
    assert timeseries_copy.columns.names == [PdFields.VARIABLE]
    # MultiIndex does not support iterating as a NamedTuple
    # https://github.com/pandas-dev/pandas/issues/34840 but we could do it via a DataFrame.
    for location_id, variable, bucket, date in to_exclude_index:
        new_tags.add(
            taglib.ZScoreOutlier(
                date=date,
                original_observation=timeseries_copy.at[(location_id, bucket, date), variable],
            ),
            location_id=location_id,
            variable=variable,
            bucket=bucket,
        )
        timeseries_copy.at[(location_id, bucket, date), variable] = np.nan
    return dataclasses.replace(dataset, timeseries_bucketed=timeseries_copy).append_tag_df(
        new_tags.as_dataframe()
    )
