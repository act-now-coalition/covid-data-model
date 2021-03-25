import enum
from dataclasses import dataclass
from typing import Collection

import more_itertools
import pandas as pd
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields
from covidactnow.datapublic.common_fields import ValueAsStrMixin
from pandas.core.dtypes.common import is_numeric_dtype

from libs import pipeline
from libs.datasets import timeseries
from libs.datasets.tail_filter import TagType

LEVEL = FieldName("level")
VARIABLE_GROUP = FieldName("variable_group")


def _location_id_to_agg(loc_id):
    """Turns a location_id into a label used for aggregation. For now this is only the
    AggregationLevel but future UI changes could let the user aggregate regions by state etc."""
    region = pipeline.Region.from_location_id(loc_id)
    return region.level.value


def _location_id_to_agg_and_state(loc_id):
    region = pipeline.Region.from_location_id(loc_id)
    if region.is_county():
        return region.state
    else:
        return region.level.value


@enum.unique
class RegionAggregationMethod(ValueAsStrMixin, str, enum.Enum):
    LEVEL = "level"
    LEVEL_AND_COUNTY_BY_STATE = "level_and_county_by_state"


@enum.unique
class VariableAggregationMethod(ValueAsStrMixin, str, enum.Enum):
    FIELD_GROUP = "field_group"
    NONE = "none"


def _get_index_level_as_series(df: pd.DataFrame, level: FieldName) -> pd.Series:
    return pd.Series(df.index.get_level_values(level), index=df.index, name=level)


def _agg_counts(
    stats_df: pd.DataFrame,
    location_id_group_by: RegionAggregationMethod,
    var_group_by: VariableAggregationMethod,
) -> pd.DataFrame:
    """Aggregate counts to make a smaller table."""
    assert stats_df.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]

    groupby = []
    location_id_series = _get_index_level_as_series(stats_df, CommonFields.LOCATION_ID)
    if location_id_group_by == RegionAggregationMethod.LEVEL:
        groupby.append(location_id_series.map(_location_id_to_agg).rename(LEVEL))
    elif location_id_group_by == RegionAggregationMethod.LEVEL_AND_COUNTY_BY_STATE:
        groupby.append(location_id_series.map(_location_id_to_agg_and_state).rename(LEVEL))
    else:
        raise ValueError("Bad location_id_group_by")

    if var_group_by == VariableAggregationMethod.FIELD_GROUP:
        variable_series = _get_index_level_as_series(stats_df, PdFields.VARIABLE)
        groupby.append(
            variable_series.map(common_fields.COMMON_FIELD_TO_GROUP).rename(VARIABLE_GROUP)
        )
    else:
        # Add variable to groupby to prevent aggregation across `variable` values.
        groupby.append(PdFields.VARIABLE)

    agg_counts = stats_df.groupby(groupby, as_index=True).sum()

    return agg_counts


@enum.unique
class StatName(ValueAsStrMixin, str, enum.Enum):
    # Count of timeseries
    HAS_TIMESERIES = "has_timeseries"
    # Count of URLs
    HAS_URL = "has_url"
    ANNOTATION_COUNT = "annotation_count"


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class AggregatedStats:
    """Aggregated statistics, where index are regions and columns are variables. Either axis may
    be filtered to keep only a subset and/or aggregated."""

    stats: pd.DataFrame

    def __post_init__(self):
        # index level 0 is a location_id or some kind of aggregated region kind of thing
        assert self.stats.index.names[0] in [CommonFields.LOCATION_ID, LEVEL]
        # index level 1 is a variable (cases, deaths, ...) or some kind of aggregated variable
        assert self.stats.index.names[1] in [PdFields.VARIABLE, VARIABLE_GROUP]
        assert self.stats.columns.to_list() == [
            StatName.HAS_TIMESERIES,
            StatName.HAS_URL,
            StatName.ANNOTATION_COUNT,
        ]
        assert is_numeric_dtype(more_itertools.one(set(self.stats.dtypes)))

    @property
    def has_timeseries(self):
        """DataFrame with column per VARIABLE or VARIABLE_GROUP"""
        return self.stats.loc(axis=1)[StatName.HAS_TIMESERIES].unstack(1)

    @property
    def has_url(self):
        """DataFrame with column per VARIABLE or VARIABLE_GROUP"""
        return self.stats.loc(axis=1)[StatName.HAS_URL].unstack(1)

    @property
    def annotation_count(self):
        """DataFrame with column per VARIABLE or VARIABLE_GROUP"""
        return self.stats.loc(axis=1)[StatName.ANNOTATION_COUNT].unstack(1)


def _xs_or_empty(df: pd.DataFrame, key: Collection[str], level: str) -> pd.DataFrame:
    """Similar to df.xs(key, level=level) but returns an empty DataFrame when key is not present"""
    mask = df.index.get_level_values(level).isin(key)
    return df.loc(axis=0)[mask]


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class PerRegionStats(AggregatedStats):
    """Instances of AggregatedStats where each row represents one timeseries."""

    def __post_init__(self):
        assert self.stats.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]

    @staticmethod
    def make(ds: timeseries.MultiRegionDataset) -> "PerRegionStats":
        # TODO(tom): Change to timeseries_bucketed
        all_timeseries_index = ds.timeseries_not_bucketed_wide_dates.index
        has_timeseries = (
            ds.timeseries_not_bucketed_wide_dates.notnull()
            .any(1)
            .astype(int)
            .reindex(index=all_timeseries_index, fill_value=False)
        )
        has_url = (
            ds.tag_all_bucket.loc[:, :, TagType.SOURCE_URL]
            .notnull()
            .astype(int)
            .reindex(index=all_timeseries_index, fill_value=0)
        )
        annotation_count = (
            ds.tag_all_bucket.loc[:, :, timeseries.ANNOTATION_TAG_TYPES]
            .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE])
            .count()
            .reindex(index=all_timeseries_index, fill_value=0)
        )

        stats = pd.DataFrame(
            {
                StatName.HAS_TIMESERIES: has_timeseries,
                StatName.HAS_URL: has_url,
                StatName.ANNOTATION_COUNT: annotation_count,
            }
        )

        return PerRegionStats(stats=stats)

    def aggregate(
        self, regions: RegionAggregationMethod, variables: VariableAggregationMethod
    ) -> AggregatedStats:
        return AggregatedStats(stats=_agg_counts(self.stats, regions, variables))

    def subset_variables(self, variables: Collection[CommonFields]) -> "PerRegionStats":
        """Returns a new PerRegionStats with only `variables` in the columns."""
        return PerRegionStats(stats=_xs_or_empty(self.stats, variables, PdFields.VARIABLE))

    def stats_for_locations(self, location_ids: pd.Index) -> pd.DataFrame:
        """Returns a DataFrame of statistics with `location_ids` as the index."""
        assert location_ids.names == [CommonFields.LOCATION_ID]
        # The stats likely don't have a value for every region. Replace any NAs with 0 so that
        # subtracting them produces a real value.
        df = (
            self.stats.groupby(CommonFields.LOCATION_ID).sum().reindex(index=location_ids).fillna(0)
        )
        df["no_url_count"] = df[StatName.HAS_TIMESERIES] - df[StatName.HAS_URL]
        return df
