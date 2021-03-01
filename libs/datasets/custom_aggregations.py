from typing import Mapping
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import timeseries
from libs.datasets import region_aggregation

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
    dc_state_region = pipeline.Region.from_fips(DC_STATE_FIPS)
    dc_county_region = pipeline.Region.from_fips(DC_COUNTY_FIPS)

    dc_map = {dc_state_region: dc_county_region}

    # aggregate_regions only copies number columns. Extract them and re-add to the aggregated
    # dataset.
    static_excluding_numbers = dataset_in.get_regions_subset(
        [dc_county_region]
    ).static.select_dtypes(exclude="number")
    dc_county_dataset = region_aggregation.aggregate_regions(dataset_in, dc_map).add_static_values(
        static_excluding_numbers.reset_index()
    )
    dataset_without_dc_county = dataset_in.remove_regions([dc_county_region])

    return dataset_without_dc_county.append_regions(dc_county_dataset)


def aggregate_puerto_rico_from_counties(
    dataset: timeseries.MultiRegionDataset,
) -> timeseries.MultiRegionDataset:
    """Returns a dataset with NA static values for the state PR aggregated from counties."""
    pr_county_mask = (dataset.static[CommonFields.STATE] == "PR") & (
        dataset.static[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
    )
    if not pr_county_mask.any():
        return dataset
    pr_counties = dataset.static.loc[pr_county_mask]
    aggregated = _aggregate_ignoring_nas(pr_counties.select_dtypes(include="number"))
    pr_location_id = pipeline.Region.from_state("PR").location_id

    patched_static = dataset.static.copy()
    for field, aggregated_value in aggregated.items():
        if pd.isna(patched_static.at[pr_location_id, field]):
            patched_static.at[pr_location_id, field] = aggregated_value

    return dataclasses.replace(dataset, static=patched_static)


def _aggregate_ignoring_nas(df_in: pd.DataFrame) -> Mapping:
    aggregated = {}
    for field in df_in.columns:
        if field == CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE:
            licensed_beds = df_in[CommonFields.LICENSED_BEDS]
            occupancy_rates = df_in[CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE]
            aggregated[field] = (licensed_beds * occupancy_rates).sum() / licensed_beds.sum()
        elif field == CommonFields.ICU_TYPICAL_OCCUPANCY_RATE:
            icu_beds = df_in[CommonFields.ICU_BEDS]
            occupancy_rates = df_in[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE]
            aggregated[field] = (icu_beds * occupancy_rates).sum() / icu_beds.sum()
        else:
            aggregated[field] = df_in[field].sum()
    return aggregated
