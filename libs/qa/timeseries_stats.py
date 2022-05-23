import enum
from dataclasses import dataclass
from typing import Collection
from typing import Sequence

import more_itertools
import pandas as pd
from datapublic import common_fields
from datapublic.common_fields import CommonFields
from datapublic.common_fields import DemographicBucket
from datapublic.common_fields import FieldName
from datapublic.common_fields import PdFields
from datapublic.common_fields import ValueAsStrMixin
from pandas.core.dtypes.common import is_numeric_dtype

from libs.datasets import AggregationLevel
from libs.datasets import dataset_utils
from libs.datasets import demographics
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets.tail_filter import TagType

DISTRIBUTION = FieldName("distribution")
FIELD_GROUP = FieldName("field_group")
SOURCE_TYPE_SET = FieldName("source_type_set")


def _get_index_level_as_series(df: pd.DataFrame, level: FieldName) -> pd.Series:
    return pd.Series(df.index.get_level_values(level), index=df.index, name=level)


@enum.unique
class StatName(ValueAsStrMixin, str, enum.Enum):
    # Count of timeseries
    HAS_TIMESERIES = "has_timeseries"
    ANNOTATION_COUNT = "annotation_count"
    BUCKET_ALL_COUNT = "bucket_all_count"
    OBSERVATION_COUNT = "observation_count"
    # Count of each tag type
    CUMULATIVE_TAIL_TRUNCATED = TagType.CUMULATIVE_TAIL_TRUNCATED
    CUMULATIVE_LONG_TAIL_TRUNCATED = TagType.CUMULATIVE_LONG_TAIL_TRUNCATED
    ZSCORE_OUTLIER = TagType.ZSCORE_OUTLIER
    KNOWN_ISSUE = TagType.KNOWN_ISSUE
    KNOWN_ISSUE_NO_DATE = TagType.KNOWN_ISSUE_NO_DATE
    KNOWN_ISSUE_DATE_RANGE = TagType.KNOWN_ISSUE_DATE_RANGE
    DERIVED = TagType.DERIVED
    DROP_FUTURE_OBSERVATION = TagType.DROP_FUTURE_OBSERVATION
    PROVENANCE = TagType.PROVENANCE
    SOURCE_URL = TagType.SOURCE_URL
    SOURCE = TagType.SOURCE


_PER_TIMESERIES_INDEX_LEVELS = [
    CommonFields.LOCATION_ID,
    PdFields.VARIABLE,
    PdFields.DEMOGRAPHIC_BUCKET,
    DISTRIBUTION,
    CommonFields.AGGREGATE_LEVEL,
    CommonFields.STATE,
    FIELD_GROUP,
]


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class Aggregated:
    """Aggregated statistics, optionally grouped by region and variable."""

    # Numeric statistics
    stats: pd.DataFrame
    # A DataFrame containing the source type as a string, or an empty DataFrame
    # TODO(tom): Change this to make more sense in aggregated instances, perhaps keeping the
    #  source types in a set.
    source_type: pd.DataFrame

    def __post_init__(self):
        # index level 0 is a location_id or some kind of aggregated region kind of thing
        assert set(self.stats.index.names).issubset(_PER_TIMESERIES_INDEX_LEVELS)
        assert self.stats.columns.to_list() == list(StatName)
        assert is_numeric_dtype(more_itertools.one(set(self.stats.dtypes)))
        assert self.source_type.index.equals(self.stats.index)
        assert self.source_type.columns.to_list() in ([SOURCE_TYPE_SET], [])

    @property
    def pivottable_data(self) -> pd.DataFrame:
        df = pd.concat([self.stats, self.source_type], axis=1).reset_index()
        return [df.columns.tolist()] + df.values.tolist()


def _xs_or_empty(df: pd.DataFrame, key: Collection[str], level: str) -> pd.DataFrame:
    """Similar to df.xs(key, level=level) but returns an empty DataFrame when key is not present"""
    mask = df.index.get_level_values(level).isin(key)
    return df.loc(axis=0)[mask]


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class PerTimeseries(Aggregated):
    """Instances of AggregatedStats where each row represents one timeseries. The index has many
    levels that are used by various groupby operations."""

    dataset: timeseries.MultiRegionDataset

    def __post_init__(self):
        super().__post_init__()
        assert self.stats.index.names == _PER_TIMESERIES_INDEX_LEVELS

    @staticmethod
    def make(ds: timeseries.MultiRegionDataset) -> "PerTimeseries":
        all_timeseries_index = ds.timeseries_bucketed_wide_dates.index

        stat_map = {}
        # These pd.Series need to have dtype int so that groupby sum doesn't turn them into a float.
        # For unknown reasons a bool is turned into a float.
        stat_map[StatName.HAS_TIMESERIES] = (
            ds.timeseries_bucketed_wide_dates.notnull()
            .any(1)
            .astype(int)
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        stat_map[StatName.ANNOTATION_COUNT] = (
            ds.tag.loc[:, :, :, timeseries.ANNOTATION_TAG_TYPES]
            .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET])
            .count()
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        stat_map[StatName.BUCKET_ALL_COUNT] = (
            _get_index_level_as_series(
                ds.timeseries_bucketed_wide_dates, PdFields.DEMOGRAPHIC_BUCKET
            )
            == DemographicBucket.ALL
        ).astype(int)
        stat_map[StatName.OBSERVATION_COUNT] = (
            ds.timeseries_bucketed_wide_dates.notnull()
            .sum(1)
            .astype(int)
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        tag_count = (
            # This groupby(...).count() could be replaced by `tag.index.value_counts(sort=False)`
            # but value_count drops the index names.
            ds.tag.groupby(
                [
                    CommonFields.LOCATION_ID,
                    PdFields.VARIABLE,
                    PdFields.DEMOGRAPHIC_BUCKET,
                    taglib.TagField.TYPE,
                ]
            )
            .count()
            .unstack(taglib.TagField.TYPE, fill_value=0)
            .reindex(columns=list(taglib.TagType), fill_value=0)
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        for tag_type in taglib.TagType:
            # Not sure why lint complains but ... pylint: disable=no-member
            stat_map[StatName._value2member_map_[tag_type]] = tag_count[tag_type]
        location_id_index = all_timeseries_index.get_level_values(CommonFields.LOCATION_ID)
        stat_extra_index = {
            # This is similar to timeseries._add_distribution_level but this operates on an index.
            DISTRIBUTION: all_timeseries_index.get_level_values(PdFields.DEMOGRAPHIC_BUCKET).map(
                lambda b: demographics.DistributionBucket.from_str(b).distribution
            ),
            CommonFields.AGGREGATE_LEVEL: location_id_index.map(
                dataset_utils.get_geo_data()[CommonFields.AGGREGATE_LEVEL]
            ),
            CommonFields.STATE: location_id_index.map(
                dataset_utils.get_geo_data()[CommonFields.STATE]
            ),
            FIELD_GROUP: all_timeseries_index.get_level_values(PdFields.VARIABLE).map(
                common_fields.COMMON_FIELD_TO_GROUP
            ),
        }
        stats = pd.DataFrame({**stat_map, **stat_extra_index}).set_index(
            list(stat_extra_index.keys()), append=True
        )
        # The source type(s) of each time series, as a string that will be identical for time
        # series with the same set of source types.
        source_type_set = (
            ds.tag_objects_series.loc(axis=0)[:, :, :, TagType.SOURCE]
            .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET])
            .apply(lambda sources: ";".join(sorted(set(s.type for s in sources))))
            .reindex(index=all_timeseries_index, fill_value="")
            .to_frame(name=SOURCE_TYPE_SET)
            .set_index(stats.index)
        )

        return PerTimeseries(stats=stats, source_type=source_type_set, dataset=ds)

    def _subset(self, field: FieldName, values: Collection) -> "PerTimeseries":
        assert field in self.stats.index.names
        return PerTimeseries(
            stats=_xs_or_empty(self.stats, values, field),
            source_type=_xs_or_empty(self.source_type, values, field),
            dataset=self.dataset,
        )

    def subset_variables(self, variables: Collection[CommonFields]) -> "PerTimeseries":
        return self._subset(PdFields.VARIABLE, variables)

    def subset_locations(self, *, aggregation_level: AggregationLevel) -> "PerTimeseries":
        return self._subset(CommonFields.AGGREGATE_LEVEL, [aggregation_level.value])

    def aggregate(self, *fields: Sequence[FieldName]) -> Aggregated:
        # Make sure fields is a list. aggregate(a, b) produces a tuple which trips up groupby.
        fields = list(fields)
        return Aggregated(
            stats=self.stats.groupby(fields, as_index=True).sum(),
            source_type=self.source_type.groupby(fields, as_index=True).sum(),
        )

    def stats_for_locations(self, location_ids: pd.Index) -> pd.DataFrame:
        """Returns a DataFrame of statistics with `location_ids` as the index."""
        assert location_ids.names == [CommonFields.LOCATION_ID]
        aggregated_by_location = self.aggregate(CommonFields.LOCATION_ID)
        # The stats likely don't have a value for every region. Replace any NAs with 0 so that
        # subtracting them produces a real value.
        df = aggregated_by_location.stats.reindex(index=location_ids).fillna(0)
        df["bucket_not_all"] = df[StatName.HAS_TIMESERIES] - df[StatName.BUCKET_ALL_COUNT]
        return df
