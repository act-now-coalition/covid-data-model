import pytest

from libs import test_positivity
from libs.pipeline import Region
from libs.pipelines import api_v2_pipeline
from libs.datasets import combined_datasets
from libs.datasets import AggregationLevel
from libs.datasets.timeseries import OneRegionTimeseriesDataset


@pytest.fixture
def nyc_regional_input(nyc_region, rt_dataset, icu_dataset):
    regional_data = combined_datasets.load_us_timeseries_dataset().get_regions_subset([nyc_region])
    test_positivity_results = test_positivity.AllMethods.run(regional_data)
    regional_data = regional_data.join_columns(
        test_positivity_results.test_positivity
    ).get_one_region(nyc_region)
    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        nyc_region, regional_data, rt_dataset, icu_dataset
    )


@pytest.fixture
def il_regional_input(rt_dataset, icu_dataset):
    region = Region.from_state("IL")
    regional_data = combined_datasets.load_us_timeseries_dataset().get_regions_subset([region])
    test_positivity_results = test_positivity.AllMethods.run(regional_data)
    regional_data = regional_data.join_columns(
        test_positivity_results.test_positivity
    ).get_one_region(region)
    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        region, regional_data, rt_dataset, icu_dataset
    )


def test_build_timeseries_and_summary_outputs(nyc_regional_input):
    timeseries = api_v2_pipeline.build_timeseries_for_region(nyc_regional_input)
    assert timeseries


def test_build_timeseries_and_summary_outputs_for_il_state(il_regional_input):
    timeseries = api_v2_pipeline.build_timeseries_for_region(il_regional_input)
    assert timeseries


def test_build_api_output_for_intervention(nyc_regional_input, tmp_path):
    county_output = tmp_path
    all_timeseries_api = api_v2_pipeline.run_on_regions([nyc_regional_input])

    api_v2_pipeline.deploy_single_level(all_timeseries_api, AggregationLevel.COUNTY, county_output)
    expected_outputs = [
        "counties.timeseries.json",
        "counties.csv",
        "counties.timeseries.csv",
        "counties.json",
        "county/36061.json",
        "county/36061.timeseries.json",
    ]

    output_paths = [
        str(path.relative_to(tmp_path)) for path in tmp_path.glob("**/*") if not path.is_dir()
    ]
    assert set(output_paths) == set(expected_outputs)


def test_output_no_timeseries_rows(nyc_regional_input, tmp_path):

    # Creating a new regional input with an empty timeseries dataset
    timeseries = nyc_regional_input.timeseries
    timeseries_data = timeseries.data.loc[timeseries.data.fips.isna()]
    regional_data = combined_datasets.RegionalData(
        nyc_regional_input.region,
        OneRegionTimeseriesDataset(timeseries_data, nyc_regional_input.latest),
    )
    regional_input = api_v2_pipeline.RegionalInput(
        nyc_regional_input.region, regional_data, None, None
    )
    assert regional_input.timeseries.empty

    all_timeseries_api = api_v2_pipeline.run_on_regions([regional_input])

    assert all_timeseries_api
