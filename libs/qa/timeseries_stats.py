import enum
from dataclasses import dataclass
from typing import Collection

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

from libs.datasets import dataset_utils
from libs.datasets import demographics
from libs.datasets import timeseries
from libs.datasets.tail_filter import TagType

DISTRIBUTION = FieldName("distribution")
FIELD_GROUP = FieldName("field_group")


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


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class Aggregated:
    """Aggregated statistics grouped by region and variable, or collections of them."""

    stats: pd.DataFrame

    def __post_init__(self):
        # index level 0 is a location_id or some kind of aggregated region kind of thing
        assert self.stats.index.names[0] in [CommonFields.LOCATION_ID, CommonFields.AGGREGATE_LEVEL]
        # index level 1 is a variable (cases, deaths, ...) or some kind of aggregated variable
        assert self.stats.index.names[1] in [PdFields.VARIABLE, FIELD_GROUP, DISTRIBUTION]
        assert self.stats.columns.to_list() == [
            StatName.HAS_TIMESERIES,
            StatName.HAS_URL,
            StatName.ANNOTATION_COUNT,
            StatName.BUCKET_ALL_COUNT,
        ]
        assert is_numeric_dtype(more_itertools.one(set(self.stats.dtypes)))

    @cached_property
    def stats_by_region_variable(self) -> pd.DataFrame:
        """A DataFrame with location index and column levels CommonField and StatName"""
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
            .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE])
            .count()
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        stat_map[StatName.BUCKET_ALL_COUNT] = (
            _get_index_level_as_series(
                ds.timeseries_bucketed_wide_dates, PdFields.DEMOGRAPHIC_BUCKET
            )
            == DemographicBucket.ALL
        ).astype(int)
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

        return PerTimeseries(stats=stats)

    def subset_variables(self, variables: Collection[CommonFields]) -> "PerTimeseries":
        return PerTimeseries(stats=_xs_or_empty(self.stats, variables, PdFields.VARIABLE))

    def aggregate(self, index: FieldName, columns: FieldName) -> Aggregated:
        return Aggregated(stats=self.stats.groupby([index, columns], as_index=True).sum())

    def stats_for_locations(self, location_ids: pd.Index) -> pd.DataFrame:
        """Returns a DataFrame of statistics with `location_ids` as the index."""
        assert location_ids.names == [CommonFields.LOCATION_ID]
        # The stats likely don't have a value for every region. Replace any NAs with 0 so that
        # subtracting them produces a real value.
        df = (
            self.stats.groupby(CommonFields.LOCATION_ID).sum().reindex(index=location_ids).fillna(0)
        )
        df["no_url_count"] = df[StatName.HAS_TIMESERIES] - df[StatName.HAS_URL]
        df["bucket_not_all"] = df[StatName.HAS_TIMESERIES] - df[StatName.BUCKET_ALL_COUNT]
        return df
