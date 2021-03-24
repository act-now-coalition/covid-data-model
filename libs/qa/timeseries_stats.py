import enum
from dataclasses import dataclass
from typing import Collection

import more_itertools
import numpy as np
import pandas as pd
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields
from covidactnow.datapublic.common_fields import ValueAsStrMixin
from pandas.core.dtypes.common import is_numeric_dtype

from libs import pipeline
from libs.datasets import timeseries
from libs.datasets.taglib import TagField
from libs.datasets.tail_filter import TagType


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


def _agg_wide_var_counts(
    wide_vars: pd.DataFrame,
    location_id_group_by: RegionAggregationMethod,
    var_group_by: VariableAggregationMethod,
) -> pd.DataFrame:
    """Aggregate wide variable counts to make a smaller table."""
    assert wide_vars.index.names == [CommonFields.LOCATION_ID]
    assert wide_vars.columns.names == [PdFields.VARIABLE]
    assert is_numeric_dtype(more_itertools.one(set(wide_vars.dtypes)))

    if location_id_group_by == RegionAggregationMethod.LEVEL:
        axis0_groupby = wide_vars.groupby(_location_id_to_agg)
    elif location_id_group_by == RegionAggregationMethod.LEVEL_AND_COUNTY_BY_STATE:
        axis0_groupby = wide_vars.groupby(_location_id_to_agg_and_state)
    else:
        raise ValueError("Bad location_id_group_by")

    agg_counts = axis0_groupby.sum().rename_axis(index=CommonFields.AGGREGATE_LEVEL)

    if var_group_by == VariableAggregationMethod.FIELD_GROUP:
        agg_counts = agg_counts.groupby(
            common_fields.COMMON_FIELD_TO_GROUP, axis=1, sort=False
        ).sum()
        # Reindex columns to match order of FieldGroup enum.
        agg_counts = agg_counts.reindex(
            columns=pd.Index(common_fields.FieldGroup).intersection(agg_counts.columns)
        )

    return agg_counts


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class AggregatedStats:
    """Aggregated statistics, where index are regions and columns are variables. Either axis may
    be filtered to keep only a subset and/or aggregated."""

    # TODO(tom): Move all these into one DataFrame so one vector operation can apply to all of them.

    # A table of count of timeseries
    has_timeseries: pd.DataFrame
    # A table of count of URLs
    has_url: pd.DataFrame
    # A table of count of annotations
    annotation_count: pd.DataFrame


@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class PerRegionStats(AggregatedStats):
    """Instances of AggregatedStats where each row represents one region."""

    @staticmethod
    def make(ds: timeseries.MultiRegionDataset) -> "PerRegionStats":
        wide_var_has_timeseries = (
            ds.timeseries_not_bucketed_wide_dates.notnull()
            .any(1)
            .unstack(PdFields.VARIABLE, fill_value=False)
            .astype(bool)
        )
        wide_var_has_url = (
            ds.tag_all_bucket.loc[:, :, TagType.SOURCE_URL].unstack(PdFields.VARIABLE).notnull()
        )
        # Need to use pivot_table instead of unstack to aggregate using sum.
        wide_var_annotation_count = pd.pivot_table(
            ds.tag_all_bucket.loc[:, :, timeseries.ANNOTATION_TAG_TYPES].notnull().reset_index(),
            values=TagField.CONTENT,
            index=CommonFields.LOCATION_ID,
            columns=PdFields.VARIABLE,
            aggfunc=np.sum,
            fill_value=0,
        )

        return PerRegionStats(
            has_timeseries=wide_var_has_timeseries,
            has_url=wide_var_has_url,
            annotation_count=wide_var_annotation_count,
        )

    def aggregate(
        self, regions: RegionAggregationMethod, variables: VariableAggregationMethod
    ) -> AggregatedStats:
        return AggregatedStats(
            has_timeseries=_agg_wide_var_counts(self.has_timeseries, regions, variables),
            has_url=_agg_wide_var_counts(self.has_url, regions, variables),
            annotation_count=_agg_wide_var_counts(self.annotation_count, regions, variables),
        )

    def subset_variables(self, variables: Collection[CommonFields]) -> "PerRegionStats":
        """Returns a new PerRegionStats with only `variables` in the columns."""
        return PerRegionStats(
            has_timeseries=self.has_timeseries.loc[
                :, self.has_timeseries.columns.intersection(variables).rename(PdFields.VARIABLE)
            ],
            has_url=self.has_url.loc[
                :, self.has_url.columns.intersection(variables).rename(PdFields.VARIABLE)
            ],
            annotation_count=self.annotation_count.loc[
                :, self.annotation_count.columns.intersection(variables).rename(PdFields.VARIABLE)
            ],
        )

    def stats_for_locations(self, location_ids: pd.Index) -> pd.DataFrame:
        """Returns a DataFrame of statistics with `location_ids` as the index."""
        assert location_ids.names == [CommonFields.LOCATION_ID]
        # The stats likely don't have a value for every region. Replace any NAs with 0 so that
        # subtracting them produces a real value.
        df = pd.DataFrame(
            {
                "annotation_count": self.annotation_count.sum(axis=1),
                "url_count": self.has_url.sum(axis=1),
                "timeseries_count": self.has_timeseries.sum(axis=1),
            },
            index=location_ids,
        ).fillna(0)
        df["no_url_count"] = df["timeseries_count"] - df["url_count"]
        return df
