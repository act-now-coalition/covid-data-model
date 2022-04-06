from dataclasses import dataclass
from typing import List, Optional, Mapping, Sequence

from datapublic.common_fields import CommonFields
from datapublic.common_fields import FieldName
from datapublic.common_fields import PdFields

import pandas as pd
import structlog
from libs import pipeline

from libs.datasets import timeseries
from libs.pipeline import Region

_log = structlog.get_logger()

MultiRegionDataset = timeseries.MultiRegionDataset


# Column for the aggregated location_id
LOCATION_ID_AGG = "location_id_agg"

FIELDS_NOT_TO_AGGREGATE = [
    # There's no way to meaningfully aggregate the raw CDC community levels across regions. So
    # they'll only be available at the county-level. (But we'll calculate our own community level
    # for all regions later in the pipeline.)
    CommonFields.CDC_COMMUNITY_LEVEL,
    CommonFields.HSA,
    CommonFields.HSA_POPULATION,
]


@dataclass(frozen=True)
class StaticWeightedAverageAggregation:
    """Represents an an average of `field` with static weights in `scale_field`."""

    # field/column/metric that gets aggregated using a weighted average
    field: FieldName
    # static field that used to produce the weights
    scale_factor: FieldName


WEIGHTED_AGGREGATIONS = (
    # Maybe test_positivity is better averaged using time-varying total tests, but it isn't
    # implemented. See TODO next to call to _find_scale_factors.
    StaticWeightedAverageAggregation(CommonFields.TEST_POSITIVITY, CommonFields.POPULATION),
    StaticWeightedAverageAggregation(CommonFields.TEST_POSITIVITY_7D, CommonFields.POPULATION),
    StaticWeightedAverageAggregation(CommonFields.TEST_POSITIVITY_14D, CommonFields.POPULATION),
    StaticWeightedAverageAggregation(
        CommonFields.VACCINATIONS_INITIATED_PCT, CommonFields.POPULATION
    ),
    StaticWeightedAverageAggregation(
        CommonFields.VACCINATIONS_COMPLETED_PCT, CommonFields.POPULATION
    ),
    StaticWeightedAverageAggregation(
        CommonFields.VACCINATIONS_ADDITIONAL_DOSE_PCT, CommonFields.POPULATION
    ),
    StaticWeightedAverageAggregation(
        CommonFields.BEDS_WITH_COVID_PATIENTS_RATIO_HSA, CommonFields.POPULATION
    ),
    StaticWeightedAverageAggregation(
        CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID_PER_100K_HSA, CommonFields.POPULATION
    ),
)


def aggregate_regions(
    dataset_in: MultiRegionDataset,
    aggregate_map: Mapping[Region, Region],
    aggregations: Sequence[StaticWeightedAverageAggregation] = WEIGHTED_AGGREGATIONS,
    *,
    reporting_ratio_required_to_aggregate: Optional[float] = None,
) -> MultiRegionDataset:
    """Produces a dataset with dataset_in aggregated using sum or weighted aggregation.

    Args:
        dataset_in: Input dataset.
        aggregate_map: Region mapping input region to aggregate region.
        aggregations: Sequence of aggregation overrides to apply aggregations other
            than sum to fields.
        reporting_ratio_required_to_aggregate: Ratio of locations per aggregate region required
            to compute aggregate value for individual data points. Uses population to weight
            ratio.

    Returns: Dataset with values aggregated to aggregate regions.
    """
    assert (
        reporting_ratio_required_to_aggregate is None
        or 0 < reporting_ratio_required_to_aggregate <= 1.0
    )
    dataset_in = dataset_in.get_regions_subset(aggregate_map.keys())
    location_id_map = {
        region_in.location_id: region_agg.location_id
        for region_in, region_agg in aggregate_map.items()
    }

    scale_fields = {agg.scale_factor for agg in aggregations}
    scaled_fields = {agg.field for agg in aggregations}
    agg_common_fields = scale_fields.intersection(scaled_fields)
    # Check that a field is not both scaled and used as the scale factor. While that
    # could make sense it isn't implemented.
    if agg_common_fields:
        raise ValueError("field and scale_factor have values in common")
    # TODO(tom): Do something smarter with non-number columns in static. Currently they are
    # silently dropped. Functions such as aggregate_to_new_york_city manually copy non-number
    # columns.
    static_in = dataset_in.static.select_dtypes(include="number")
    scale_field_missing = scale_fields.difference(static_in.columns)
    if scale_field_missing:
        raise ValueError("Unable to do scaling due to missing column")
    # Split static_in into two DataFrames, by column:
    scale_fields_mask = static_in.columns.isin(scale_fields)
    # Static input values used to create scale factors and ...
    static_in_scale_fields = static_in.loc[:, scale_fields_mask]
    # ... all other static input values.
    static_in_other_fields = static_in.loc[:, ~scale_fields_mask]

    populations = None
    if reporting_ratio_required_to_aggregate is not None:
        populations = static_in.loc[:, CommonFields.POPULATION]

    static_agg_scale_fields = _aggregate_dataframe_by_region(
        static_in_scale_fields,
        location_id_map,
        reporting_ratio_location_weights=populations,
        reporting_ratio_required=reporting_ratio_required_to_aggregate,
    )
    location_ids = dataset_in.timeseries.index.get_level_values(CommonFields.LOCATION_ID)
    # TODO(tom): Add support for time-varying scale factors, for example to scale
    # test_positivity by number of tests.

    scale_factors = _find_scale_factors(
        aggregations,
        location_id_map,
        static_agg_scale_fields,
        static_in_scale_fields,
        location_ids,
    )

    static_other_fields_scaled = _apply_scaling_factor(
        static_in_other_fields, scale_factors, aggregations
    )
    timeseries_scaled = _apply_scaling_factor(dataset_in.timeseries, scale_factors, aggregations)

    static_agg_other_fields = _aggregate_dataframe_by_region(
        static_other_fields_scaled,
        location_id_map,
        reporting_ratio_location_weights=populations,
        reporting_ratio_required=reporting_ratio_required_to_aggregate,
    )
    timeseries_agg = _aggregate_dataframe_by_region(
        timeseries_scaled,
        location_id_map,
        reporting_ratio_location_weights=populations,
        reporting_ratio_required=reporting_ratio_required_to_aggregate,
    )
    static_agg = pd.concat([static_agg_scale_fields, static_agg_other_fields], axis=1)
    if static_agg.index.name != CommonFields.LOCATION_ID:
        # It looks like concat doesn't always set the index name, but haven't worked out
        # the pattern of when the fix is needed.
        static_agg = static_agg.rename_axis(index=CommonFields.LOCATION_ID)

    if CommonFields.AGGREGATE_LEVEL in dataset_in.static.columns:
        # location_id_to_level returns an AggregationLevel enum, but we use the str in DataFrames.
        # These are not equivalent so put the `value` attribute in static_agg.
        static_agg[CommonFields.AGGREGATE_LEVEL] = (
            static_agg.index.get_level_values(CommonFields.LOCATION_ID)
            .map(pipeline.location_id_to_level)
            .map(lambda l: l.value)
        )
    if CommonFields.FIPS in dataset_in.static.columns:
        static_agg[CommonFields.FIPS] = static_agg.index.get_level_values(
            CommonFields.LOCATION_ID
        ).map(pipeline.location_id_to_fips)

    # TODO(tom): Copy tags (annotations and provenance) to the return value.
    return MultiRegionDataset(timeseries=timeseries_agg, static=static_agg)


def _calculate_weighted_reporting_ratio(
    long_all_values: pd.Series,
    location_id_map: Mapping[str, str],
    scale_series: pd.Series,
    groupby_columns: List[str],
) -> pd.Series:
    """Calculates weighted ratio of locations reporting data scaled by `scale_series`.

    Args:
        long_all_values: All values as a series
        location_id_map: Map of input region location_id to aggregate region location id.
        scale_series: Series with index of CommonFields.LOCATION_ID of weights for each
            location id. For example, population of each location id.
        groupby_columns: Columns to group scaled values by when aggregating.

    Returns: Series of scaled ratio of regions reporting with an index of `groupby_columns`.
    """
    assert long_all_values.index.names == [CommonFields.LOCATION_ID] + groupby_columns
    assert scale_series.index.names == [CommonFields.LOCATION_ID]

    scale_field = "scale"

    location_id_df = pd.DataFrame(
        location_id_map.items(), columns=[CommonFields.LOCATION_ID, LOCATION_ID_AGG]
    ).set_index(CommonFields.LOCATION_ID)
    location_id_df[scale_field] = scale_series
    location_id_aggregated_scale_field = location_id_df.groupby(LOCATION_ID_AGG)[scale_field].sum()

    long_all_scaled_count = long_all_values.notna() * scale_series
    long_agg_scaled = long_all_scaled_count.groupby(groupby_columns).sum()
    return long_agg_scaled / location_id_aggregated_scale_field


def _find_scale_factors(
    aggregations: Sequence[StaticWeightedAverageAggregation],
    location_id_map: Mapping[str, str],
    static_agg: pd.DataFrame,
    static_in: pd.DataFrame,
    location_ids: Sequence[str],
) -> pd.DataFrame:
    assert static_in.index.names == [CommonFields.LOCATION_ID]
    assert static_agg.index.names == [CommonFields.LOCATION_ID]
    # For each location_id, calculate the scaling factor from the static data.
    scale_factors = pd.DataFrame([], index=pd.Index(location_ids).unique().sort_values())
    for scale_factor_field in {agg.scale_factor for agg in aggregations}:
        if scale_factor_field in static_in.columns and scale_factor_field in static_agg.columns:
            # Make a series with index of the un-aggregated location_ids that has values of the
            # corresponding aggregated field value.
            agg_values = (
                static_in.index.to_series(index=static_in.index)
                .map(location_id_map)  # Maps from un-aggregated to aggregated location_id
                .map(static_agg[scale_factor_field])  # Gets the aggregated value
            )
            scale_factors[scale_factor_field] = static_in[scale_factor_field] / agg_values
    return scale_factors


def _apply_scaling_factor(
    df_in: pd.DataFrame,
    scale_factors: pd.DataFrame,
    aggregations: Sequence[StaticWeightedAverageAggregation],
) -> pd.DataFrame:
    """Returns a copy of df_in with some fields scaled according to `aggregations`.

    Args:
        df_in: Input un-aggregated timeseries or static data
        scale_factors: For each scale_factor field, the per-region scaling factor
        aggregations: Describes the fields to be scaled
        """
    assert df_in.index.names in (
        [CommonFields.LOCATION_ID, CommonFields.DATE],
        [CommonFields.LOCATION_ID],
    )
    # Check that scale_factors has location index and CommonFields in columns.
    assert scale_factors.index.names == [CommonFields.LOCATION_ID]
    assert scale_factors.columns.difference(CommonFields.list()).empty

    # Scaled fields are modified in-place
    df_out = df_in.copy()

    for agg in aggregations:
        if agg.field in df_in.columns and agg.scale_factor in scale_factors.columns:
            df_out[agg.field] = df_out[agg.field] * scale_factors[agg.scale_factor]

    return df_out


def _aggregate_dataframe_by_region(
    df_in: pd.DataFrame,
    location_id_map: Mapping[str, str],
    *,
    reporting_ratio_location_weights: Optional[pd.Series] = None,
    reporting_ratio_required: float = 1.0,
) -> pd.DataFrame:
    """Aggregates a DataFrame using given region map. The output contains dates iff the input does."""

    if CommonFields.DATE in df_in.index.names:
        groupby_columns = [LOCATION_ID_AGG, CommonFields.DATE, PdFields.VARIABLE]
        empty_result = timeseries.EMPTY_TIMESERIES_WIDE_VARIABLES_DF
    else:
        groupby_columns = [LOCATION_ID_AGG, PdFields.VARIABLE]
        empty_result = timeseries.EMPTY_STATIC_DF

    # df_in is sometimes empty in unittests. Return a DataFrame that is also empty and
    # has enough of an index that the test passes.
    if df_in.empty:
        return empty_result

    df = df_in.copy()  # Copy because the index is modified below

    # Add a new level in the MultiIndex with the new location_id_agg
    # From https://stackoverflow.com/a/56278735
    old_idx = df.index.to_frame()
    # Add location_id_agg so that when location_id is removed the remaining MultiIndex levels
    # match the levels of groupby_columns.
    old_idx.insert(1, LOCATION_ID_AGG, old_idx[CommonFields.LOCATION_ID].map(location_id_map))
    df.index = pd.MultiIndex.from_frame(old_idx)

    # Stack into a Series with several levels in the index.
    long_all_values = df.rename_axis(columns=PdFields.VARIABLE).stack(dropna=True)
    assert long_all_values.index.names == [CommonFields.LOCATION_ID] + groupby_columns

    # Aggregate by location_id_agg, optional date and variable.
    long_agg = long_all_values.groupby(groupby_columns, sort=False).sum()

    long_agg = long_agg.loc[
        ~long_agg.index.get_level_values(PdFields.VARIABLE).isin(FIELDS_NOT_TO_AGGREGATE)
    ]

    if reporting_ratio_required:
        weighted_reporting_ratio = _calculate_weighted_reporting_ratio(
            long_all_values, location_id_map, reporting_ratio_location_weights, groupby_columns
        )
        is_valid_reporting_ratio = weighted_reporting_ratio >= reporting_ratio_required
        long_agg = long_agg.loc[is_valid_reporting_ratio]
        # TODO(tom): Find a way to expose the aggregation internals. For now uncomment the
        #  following code and fire up your debugger.
        # print(long_agg.loc[~is_valid_reporting_ratio].index)
        # if CommonFields.DATE in groupby_columns:
        #     print(is_valid_reporting_ratio.index)
        #     not_aggregated = is_valid_reporting_ratio.loc[~is_valid_reporting_ratio].index
        #     # Similar to timeseries._slice_with_labels but with a DataFrame instead of Series.
        #     agg_not_enough_reporting = (
        #         long_all_values.unstack("location_id")
        #         .isna()
        #         .reset_index()
        #         .set_index(not_aggregated.names)
        #         .reindex(not_aggregated)
        #     )
        #     # A table with a row for each aggregated location-variable that didn't have enough
        #     # reporting inputs and column for each input location and True where the input was NA.
        #     print(agg_not_enough_reporting)

    df_out = (
        long_agg.unstack()
        .rename_axis(index={LOCATION_ID_AGG: CommonFields.LOCATION_ID})
        .sort_index()
        .reindex(columns=df_in.columns)
    )
    assert df_in.index.names == df_out.index.names
    return df_out
