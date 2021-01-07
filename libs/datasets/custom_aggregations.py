from libs import pipeline
from libs.datasets import timeseries

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
    nyc_dataset = timeseries.aggregate_regions(ds_in, nyc_map, ignore_na=True).add_static_values(
        static_excluding_numbers.reset_index()
    )

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
    dc_county_dataset = timeseries.aggregate_regions(
        dataset_in, dc_map, ignore_na=True
    ).add_static_values(static_excluding_numbers.reset_index())
    dataset_without_dc_county = dataset_in.remove_regions([dc_county_region])

    return dataset_without_dc_county.append_regions(dc_county_dataset)
