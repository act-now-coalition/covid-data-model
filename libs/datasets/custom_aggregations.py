from typing import Mapping
import dataclasses
import pandas as pd
import structlog

from datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import region_aggregation
from libs.datasets import AggregationLevel
from libs.datasets import timeseries
from libs.pipeline import Region

NEW_YORK_COUNTY = "New York County"
NEW_YORK_COUNTY_FIPS = "36061"
NEW_YORK_CITY_FIPS = "3651000"

NYC_BOROUGH_FIPS = [
    "36047",  # Kings County
    "36081",  # Queens
    "36005",  # Bronx
    "36085",  # Richmond
]
ALL_NYC_FIPS = NYC_BOROUGH_FIPS + [NEW_YORK_COUNTY_FIPS]
ALL_NYC_REGIONS = [pipeline.Region.from_fips(fips) for fips in ALL_NYC_FIPS]

DC_COUNTY_FIPS = "11001"
DC_STATE_FIPS = "11"


def aggregate_to_new_york_city(
    ds_in: timeseries.MultiRegionDataset,
) -> timeseries.MultiRegionDataset:
    nyc_region = pipeline.Region.from_fips(NEW_YORK_CITY_FIPS)
    # Map from borough / county to the region used for aggregated NYC
    nyc_map = {borough_region: nyc_region for borough_region in ALL_NYC_REGIONS}

    # aggregate_regions only copies number columns. Extract them and re-add to the aggregated
    # dataset.
    static_excluding_numbers = ds_in.get_regions_subset([nyc_region]).static.select_dtypes(
        exclude="number"
    )
    nyc_dataset = region_aggregation.aggregate_regions(
        ds_in, nyc_map, reporting_ratio_required_to_aggregate=None
    ).add_static_values(static_excluding_numbers.reset_index())

    return ds_in.append_regions(nyc_dataset)


def replace_dc_county_with_state_data(
    dataset_in: timeseries.MultiRegionDataset,
) -> timeseries.MultiRegionDataset:
    """Replace DC County data with data from State.

    Args:
        dataset_in: Input dataset.

    Returns: Dataset with DC county data replaced to match DC state.
    """
    import logging

    logging.info(f"in replace for {dataset_in.location_ids[0]}")
    dc_state_region = pipeline.Region.from_fips(DC_STATE_FIPS)
    dc_county_region = pipeline.Region.from_fips(DC_COUNTY_FIPS)

    dc_map = {dc_state_region: dc_county_region}

    # aggregate_regions only copies number columns. Extract them and re-add to the aggregated
    # dataset.
    dataset_with_dc_county, dataset_without_dc_county = dataset_in.partition_by_region(
        [dc_county_region]
    )
    static_excluding_numbers = dataset_with_dc_county.static.select_dtypes(exclude="number")
    dc_county_dataset = region_aggregation.aggregate_regions(dataset_in, dc_map).add_static_values(
        static_excluding_numbers.reset_index()
    )

    # The aggregation will have replaced hsa_population with nan.  We need to fix it.
    dc_county_dataset.static.loc[
        :, CommonFields.HSA_POPULATION
    ] = dataset_with_dc_county.static.loc[:, CommonFields.HSA_POPULATION]

    return dataset_without_dc_county.append_regions(dc_county_dataset)


def aggregate_puerto_rico_from_counties(
    dataset: timeseries.MultiRegionDataset,
) -> timeseries.MultiRegionDataset:
    """Returns a dataset with NA static values for the state PR aggregated from counties."""
    pr_counties = dataset.get_subset(AggregationLevel.COUNTY, state="PR")
    if pr_counties.location_ids.empty:
        return dataset
    aggregated = _aggregate_ignoring_nas(pr_counties.static.select_dtypes(include="number"))
    pr_location_id = pipeline.Region.from_state("PR").location_id

    patched_static = dataset.static.copy()
    for field, aggregated_value in aggregated.items():
        if pd.isna(patched_static.at[pr_location_id, field]):
            patched_static.at[pr_location_id, field] = aggregated_value

    return dataclasses.replace(dataset, static=patched_static)


# TODO(michael): I ripped out some code for aggregating typicalUsageRate. Is
# this function even still necessary?
def _aggregate_ignoring_nas(df_in: pd.DataFrame) -> Mapping:
    aggregated = {}
    for field in df_in.columns:
        aggregated[field] = df_in[field].sum()
    return aggregated


US_AGGREGATED_EXPECTED_VARIABLES_TO_DROP = [
    CommonFields.CASES,
    CommonFields.NEW_CASES,
    CommonFields.DEATHS,
    CommonFields.NEW_DEATHS,
    CommonFields.POPULATION,
    CommonFields.VACCINATIONS_COMPLETED,
    CommonFields.VACCINATIONS_INITIATED,
    CommonFields.VACCINES_ADMINISTERED,
    CommonFields.VACCINES_DISTRIBUTED,
]
US_AGGREGATED_VARIABLE_DROP_MESSAGE = (
    "Unexpected variable found in source and aggregated country data."
)


def _log_unexpected_aggregated_variables_to_drop(variables_to_drop: pd.Index):
    unexpected_drops = variables_to_drop.difference(US_AGGREGATED_EXPECTED_VARIABLES_TO_DROP)
    if not unexpected_drops.empty:
        log = structlog.get_logger()
        log.warn(
            US_AGGREGATED_VARIABLE_DROP_MESSAGE, variable=unexpected_drops.to_list(),
        )


def aggregate_to_country(
    dataset_in: timeseries.MultiRegionDataset, *, reporting_ratio_required_to_aggregate: float
):
    region_us = Region.from_location_id("iso1:us")
    unaggregated_us, others = dataset_in.partition_by_region([region_us])
    aggregated_us = region_aggregation.aggregate_regions(
        others,
        pipeline.us_states_and_territories_to_country_map(),
        reporting_ratio_required_to_aggregate=reporting_ratio_required_to_aggregate,
    )
    # Prioritize unaggregated over aggregated. This could be done using
    # timeseries.combined_datasets but what I'm doing here seems more straight-forward and more
    # likely to preserve tags.
    # Any variables that have both a US country level real value in the unaggregated data and
    # aggregated_us are dropped from the aggregated data.
    unaggregated_us_real = unaggregated_us.drop_na_columns()
    variables_to_drop = aggregated_us.variables.intersection(unaggregated_us_real.variables)
    _log_unexpected_aggregated_variables_to_drop(variables_to_drop)
    aggregated_us = aggregated_us.drop_columns_if_present(variables_to_drop)
    joined_us = unaggregated_us_real.join_columns(aggregated_us)
    return others.append_regions(joined_us)
