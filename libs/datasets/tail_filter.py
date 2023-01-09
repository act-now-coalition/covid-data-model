from typing import ClassVar, List, Tuple
from typing import Type

import math
import dataclasses

import pandas as pd
from datapublic.common_fields import CommonFields
from datapublic.common_fields import FieldName
from datapublic.common_fields import PdFields

from libs.datasets import taglib
from libs.datasets import timeseries

TagType = taglib.TagType
TagField = taglib.TagField


# The Series.name in apply is a tuple copied from the index. Make an assert fail when the index
# names change so we know to update access to the tuple elements.
# https://stackoverflow.com/questions/26658240/getting-the-index-of-a-row-in-a-pandas-apply-function
_EXPECTED_INDEX_NAMES = [CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET]


@dataclasses.dataclass
class TailFilter:
    _annotations: taglib.TagCollection = dataclasses.field(default_factory=taglib.TagCollection)

    # Counts that track what the filter has done.
    skipped_too_short: int = 0
    skipped_na_mean: int = 0
    all_good: int = 0
    truncated: int = 0
    long_truncated: int = 0

    # Trust values 2 - 4 weeks ago to have reasonable diffs. Access series_in.iat[-15] to
    # series_in.iat[-28] (inclusive).
    TRUSTED_DATES_OLDEST: ClassVar[int] = -28
    TRUSTED_DATES_NEWEST: ClassVar[int] = -15

    # Possibly drop series_in.iat[-1] to series_in.iat[-14] (inclusive).
    FILTER_DATES_OLDEST: ClassVar[int] = -14
    # If this many real values are dropped use the more severe TagType.
    COUNT_OBSERVATION_LONG: ClassVar[int] = 8

    @staticmethod
    def run(
        dataset: timeseries.MultiRegionDataset, fields: List[FieldName]
    ) -> Tuple["TailFilter", timeseries.MultiRegionDataset]:
        """Returns a dataset with recent data that looks bad removed from cumulative fields."""
        timeseries_wide_dates = dataset.timeseries_bucketed_wide_dates

        fields_mask = timeseries_wide_dates.index.get_level_values(PdFields.VARIABLE).isin(fields)
        to_filter = timeseries_wide_dates.loc[pd.IndexSlice[:, fields_mask], :]
        not_filtered = timeseries_wide_dates.loc[pd.IndexSlice[:, ~fields_mask], :]

        tail_filter = TailFilter()
        assert to_filter.index.names == _EXPECTED_INDEX_NAMES
        filtered = to_filter.apply(tail_filter._filter_one_series, axis=1)

        merged = pd.concat([not_filtered, filtered])
        timeseries_wide_variables = merged.stack().unstack(PdFields.VARIABLE).sort_index()

        # TODO(tom): Find a generic way to return the counts in tail_filter and stop returning the
        #  object itself.
        return (
            tail_filter,
            dataclasses.replace(
                dataset, timeseries_bucketed=timeseries_wide_variables
            ).append_tag_df(tail_filter._annotations.as_dataframe()),
        )

    def _filter_one_series(self, series_in: pd.Series) -> pd.Series:
        """Filters one timeseries of cumulative values. This is a method so self can be used to
        store side outputs.

        Args:
            series_in: a timeseries of float values, with a sorted DatetimeIndex
            """
        if len(series_in) < -TailFilter.TRUSTED_DATES_OLDEST:
            self.skipped_too_short += 1
            return series_in

        diff = series_in.diff()
        mean = diff.iloc[
            TailFilter.TRUSTED_DATES_OLDEST : TailFilter.TRUSTED_DATES_NEWEST + 1
        ].mean()
        if pd.isna(mean):
            self.skipped_na_mean += 1
            return series_in
        # Pick a bound used to determine if the diff of recent values are reasonable. For
        # now values less than 100th of the mean are considered stalled. The variance could be
        # used to create a tighter bound but the simple threshold seems to work for now.
        # TODO(tom): Experiment with other ways to calculate the bound.
        threshold = math.floor(mean / 100)

        # Go backwards in time, starting with the most recent observation. Stop as soon as there
        # is an observation with diff that looks reasonable / not stalled / over the threshold.
        # If no such diff is found after reading iat[FILTER_DATES_OLDEST] the 'else' is taken.
        count_observation_diff_under_threshold = 0
        for i in range(-1, TailFilter.FILTER_DATES_OLDEST - 1, -1):  # range stop is exclusive
            if pd.isna(diff.iat[i]):
                continue
            if diff.iat[i] >= threshold:
                truncate_at = i + 1
                break
            else:
                count_observation_diff_under_threshold += 1
        else:
            # Reached end of loop without finding a diff >= threshold.
            truncate_at = TailFilter.FILTER_DATES_OLDEST
        if count_observation_diff_under_threshold == 0:
            self.all_good += 1
            return series_in
        else:
            # series_in.iat[truncate_at] is the first value *not* returned
            assert TailFilter.FILTER_DATES_OLDEST <= truncate_at <= -1
            annotation_type: Type[taglib.AnnotationWithDate]
            if count_observation_diff_under_threshold < TailFilter.COUNT_OBSERVATION_LONG:
                annotation_type = taglib.CumulativeTailTruncated
                self.truncated += 1
            else:
                annotation_type = taglib.CumulativeLongTailTruncated
                self.long_truncated += 1
            # Currently one annotation is created per series. Maybe it makes more sense to add
            # one for each dropped observation / real value?
            # https://github.com/act-now-coalition/covid-data-model/pull/855#issuecomment-747698288
            self._annotations.add(
                annotation_type(
                    date=series_in.index[truncate_at - 1],
                    original_observation=float(series_in.iat[truncate_at]),
                ),
                # `name` is a tuple with elements _EXPECTED_INDEX_NAMES
                location_id=series_in.name[0],
                variable=series_in.name[1],
                bucket=series_in.name[2],
            )
            # TODO(tom): add count of removed observations or list of all removed or one
            #  annotation per removed observations
            # Using integer position indexing where the upper bound is exclusive, like regular
            # Python indexing and unlike Pandas label (`loc` and `at`) indexing.
            return series_in.iloc[:truncate_at]
