import enum
from dataclasses import dataclass
from typing import Collection
from typing import Sequence

import more_itertools
import pandas as pd
from backports.cached_property import cached_property
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields
from covidactnow.datapublic.common_fields import ValueAsStrMixin
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
    # Count of URLs
    HAS_URL = "has_url"
    ANNOTATION_COUNT = "annotation_count"
    BUCKET_ALL_COUNT = "bucket_all_count"
    SOURCE_TYPE_SET = "source_type_set"
    CUMULATIVE_TAIL_TRUNCATED = "cumulative_tail_truncated"
    CUMULATIVE_LONG_TAIL_TRUNCATED = "cumulative_long_tail_truncated"
    ZSCORE_OUTLIER = "zscore_outlier"
    KNOWN_ISSUE = "known_issue"
    DERIVED = "derived"
    PROVENANCE = "provenance"
    SOURCE_URL = "source_url"
    SOURCE = "source"


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
        assert self.stats.index.names[0] in [CommonFields.LOCATION_ID, CommonFields.AGGREGATE_LEVEL]
        assert self.stats.columns.to_list() == list(StatName)
        assert is_numeric_dtype(more_itertools.one(set(self.stats.dtypes)))
        assert self.source_type.index.equals(self.stats.index)
        assert self.source_type.columns.to_list() in ([SOURCE_TYPE_SET], [])

    @property
    def pivottable_data(self) -> pd.DataFrame:
        df = pd.concat([self.stats, self.source_type], axis=1).reset_index()
        return [df.columns.tolist()] + df.values.tolist()

    @cached_property
    def stats_by_region_variable(self) -> pd.DataFrame:
        """A DataFrame with location index and column levels CommonField and StatName"""
        # index level 1 is a variable (cases, deaths, ...) or some kind of aggregated variable
        assert self.stats.index.names[1] in [PdFields.VARIABLE, FIELD_GROUP, DISTRIBUTION]

        # The names of index levels 0 and 1 may vary. There doesn't seem to be a way to pass the
        # index level numbers to groupby so lookup the names.
        groupby = [self.stats.index.names[0], self.stats.index.names[1]]
        return self.stats.groupby(groupby, as_index=True).sum().unstack(1)

    @property
    def has_timeseries(self):
        """DataFrame with column per VARIABLE or FIELD_GROUP"""
        return self.stats_by_region_variable.loc(axis=1)[StatName.HAS_TIMESERIES]

    @property
    def has_url(self):
        """DataFrame with column per VARIABLE or FIELD_GROUP"""
        return self.stats_by_region_variable.loc(axis=1)[StatName.HAS_URL]

    @property
    def annotation_count(self):
        """DataFrame with column per VARIABLE or FIELD_GROUP"""
        return self.stats_by_region_variable.loc(axis=1)[StatName.ANNOTATION_COUNT]


def _xs_or_empty(df: pd.DataFrame, key: Collection[str], level: str) -> pd.DataFrame:
    """Similar to df.xs(key, level=level) but returns an empty DataFrame when key is not present"""
    mask = df.index.get_level_values(level).isin(key)
    return df.loc(axis=0)[mask]


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class PerTimeseries(Aggregated):
    """Instances of AggregatedStats where each row represents one timeseries. The index has many
    levels that are used by various groupby operations."""

    def __post_init__(self):
        super().__post_init__()
        assert self.stats.index.names == [
            CommonFields.LOCATION_ID,
            PdFields.VARIABLE,
            PdFields.DEMOGRAPHIC_BUCKET,
            DISTRIBUTION,
            CommonFields.AGGREGATE_LEVEL,
            CommonFields.STATE,
            FIELD_GROUP,
        ]

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
        stat_map[StatName.HAS_URL] = (
            ds.tag.loc[:, :, :, TagType.SOURCE_URL]
            .notnull()
            .astype(int)
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        stat_map[StatName.ANNOTATION_COUNT] = (
            ds.tag.loc[:, :, :, timeseries.ANNOTATION_TAG_TYPES]
            .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET])
            .count()
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        tag_count = (
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
            stat_map[StatName._value2member_map_[tag_type]] = tag_count[tag_type]
        location_id_index = all_timeseries_index.get_level_values(CommonFields.LOCATION_ID)
        stat_extra_index = {
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

        return PerTimeseries(stats=stats, source_type=source_type_set)

    def _subset(self, field: FieldName, values: Collection) -> "PerTimeseries":
        assert field in self.stats.index.names
        return PerTimeseries(
            stats=_xs_or_empty(self.stats, values, field),
            source_type=_xs_or_empty(self.source_type, values, field),
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
        df["no_url_count"] = df[StatName.HAS_TIMESERIES] - df[StatName.HAS_URL]
        df["bucket_not_all"] = df[StatName.HAS_TIMESERIES] - df[StatName.BUCKET_ALL_COUNT]
        return df
