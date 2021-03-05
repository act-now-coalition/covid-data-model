import dataclasses
import datetime
import pathlib
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List, Optional, Union, TextIO
from typing import Mapping
from typing import Set
from typing import Sequence
from typing import Tuple

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields
from pandas.core.dtypes.common import is_numeric_dtype
from typing_extensions import final

import pandas as pd
import numpy as np
import structlog
from covidactnow.datapublic import common_df
from libs import pipeline
from libs.datasets import dataset_pointer
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.dataset_utils import GEO_DATA_COLUMNS
from libs.datasets.dataset_utils import NON_NUMERIC_COLUMNS
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets import taglib
from libs.datasets.taglib import TagField
from libs.datasets.taglib import TagType
from libs.datasets.taglib import UrlStr
from libs.pipeline import Region
import pandas.core.groupby.generic
from backports.cached_property import cached_property
from libs.datasets import timeseries

MultiRegionDataset = timeseries.MultiRegionDataset


def _calculate_modified_zscore(
    series: pd.Series, window: int = 10, min_periods=3, ignore_zeros=True
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
    if ignore_zeros:
        series[series == 0] = None

    rolling_series = series.rolling(window=window, min_periods=min_periods)
    # Shifting one to exclude current datapoint
    mean = rolling_series.mean().shift(1)
    std = rolling_series.std(ddof=0).shift(1)
    z = (series - mean) / std
    return z.abs()


def drop_new_case_outliers(
    timeseries: MultiRegionDataset, zscore_threshold: float = 8.0, case_threshold: int = 30,
) -> MultiRegionDataset:
    """Identifies and drops outliers from the new case series.

    Args:
        timeseries: Timeseries.
        zscore_threshold: Z-score threshold.  All new cases with a zscore greater than the
            threshold will be removed.
        case_threshold: Min number of cases needed to count as an outlier.

    Returns: timeseries with outliers removed from new_cases.
    """
    return drop_series_outliers(
        timeseries,
        CommonFields.NEW_CASES,
        zscore_threshold=zscore_threshold,
        threshold=case_threshold,
    )


def drop_series_outliers(
    dataset: MultiRegionDataset,
    field: CommonFields,
    zscore_threshold: float = 8.0,
    threshold: int = 30,
) -> MultiRegionDataset:
    """Identifies and drops outliers from the new case series.

    Args:
        timeseries: Timeseries.
        zscore_threshold: Z-score threshold.  All new cases with a zscore greater than the
            threshold will be removed.
        threshold: Min number of cases needed to count as an outlier.

    Returns: timeseries with outliers removed from new_cases.
    """
    df_copy = dataset.timeseries.copy()
    grouped_df = dataset.groupby_region()

    zscores = grouped_df[field].apply(_calculate_modified_zscore)
    to_exclude = (zscores > zscore_threshold) & (df_copy[field] > threshold)

    new_tags = taglib.TagCollection()
    # to_exclude is a Series of bools with the same index as df_copy. Iterate through the index
    # rows where to_exclude is True.
    assert to_exclude.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
    values = [(idx, df_copy.at[idx, field]) for idx in to_exclude[to_exclude].keys()]
    for (location_id, date), original_value in values:
        tag = taglib.ZScoreOutlier(date=date, original_observation=original_value,)
        new_tags.add(tag, location_id=location_id, variable=field)
    df_copy.loc[to_exclude, field] = np.nan

    new_dataset = dataclasses.replace(dataset, timeseries=df_copy).append_tag_df(
        new_tags.as_dataframe()
    )

    return new_dataset


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
    df_copy = dataset.timeseries.copy()
    grouped_df = dataset.groupby_region()

    def process_series(series):
        recent_series = series.tail(10)
        # window of 5 days seems to capture about the right amount of variance.
        # If window is too large, there may have been a large enough natural shift
        # in test positivity that recent extreme value looks more noraml.
        series = _calculate_modified_zscore(recent_series, window=5, ignore_zeros=False)
        series.index = series.index.get_level_values(CommonFields.DATE)
        series = series.dropna().last("1D")
        return series

    zscores = grouped_df[CommonFields.TEST_POSITIVITY_7D].apply(process_series)
    test_positivity_diffs = df_copy[CommonFields.TEST_POSITIVITY_7D].diff().abs()

    to_exclude = (zscores > zscore_threshold) & (test_positivity_diffs > diff_threshold_ratio)

    new_tags = taglib.TagCollection()
    # to_exclude is a Series of bools with the same index as df_copy. Iterate through the index
    # rows where to_exclude is True.
    assert to_exclude.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
    for idx, _ in to_exclude[to_exclude].iteritems():
        new_tags.add(
            taglib.ZScoreOutlier(
                date=idx[1], original_observation=df_copy.at[idx, CommonFields.TEST_POSITIVITY_7D],
            ),
            location_id=idx[0],
            variable=CommonFields.TEST_POSITIVITY_7D,
        )

    df_copy.loc[to_exclude[to_exclude].index, CommonFields.TEST_POSITIVITY_7D] = np.nan

    return dataclasses.replace(dataset, timeseries=df_copy).append_tag_df(new_tags.as_dataframe())
