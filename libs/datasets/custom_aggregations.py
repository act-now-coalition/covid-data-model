from libs import pipeline
from libs.datasets import timeseries
from libs.datasets.dataset_utils import AggregationLevel

NEW_YORK_COUNTY = "New York County"
NEW_YORK_COUNTY_FIPS = "36061"

NYC_BOROUGH_FIPS = [
    "36047",  # Kings County
    "36081",  # Queens
    "36005",  # Bronx
    "36085",  # Richmond
]
ALL_NYC_FIPS = NYC_BOROUGH_FIPS + [NEW_YORK_COUNTY_FIPS]


def aggregate_to_new_york_city(
    ds_in: timeseries.MultiRegionDataset,
) -> timeseries.MultiRegionDataset:
    nyc_region = pipeline.Region.from_fips(NEW_YORK_COUNTY_FIPS)
    all_nyc_regions = [pipeline.Region.from_fips(fips) for fips in ALL_NYC_FIPS]
    nyc_map = {borough_region: nyc_region for borough_region in all_nyc_regions}

    # aggregate_regions only copies number columns. Extract them and re-add to the aggregated
    # dataset.
    static_excluding_numbers = ds_in.get_regions_subset([nyc_region]).static.select_dtypes(
        exclude="number"
    )
    nyc_dataset = timeseries.aggregate_regions(
        ds_in, nyc_map, AggregationLevel.COUNTY, ignore_na=True
    ).add_static_values(static_excluding_numbers.reset_index())

    return ds_in.remove_regions(all_nyc_regions).append_regions(nyc_dataset)
