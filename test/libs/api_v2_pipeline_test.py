from libs.pipelines import api_v2_pipeline
from libs.datasets import combined_datasets
from libs.datasets import AggregationLevel
from libs.datasets.timeseries import OneRegionTimeseriesDataset


def test_build_timeseries_and_summary_outputs(nyc_model_output_path, nyc_region):

    regional_input = api_v2_pipeline.RegionalInput.from_region_and_model_output(
        nyc_region, nyc_model_output_path.parent
    )
    timeseries = api_v2_pipeline.build_timeseries_for_region(regional_input)

    assert timeseries


def test_build_api_output_for_intervention(nyc_region, nyc_model_output_path, tmp_path):
    county_output = tmp_path
    regional_input = api_v2_pipeline.RegionalInput.from_region_and_model_output(
        nyc_region, nyc_model_output_path.parent
    )
    all_timeseries_api = api_v2_pipeline.run_on_regions([regional_input])

    api_v2_pipeline.deploy_single_level(all_timeseries_api, AggregationLevel.COUNTY, county_output)
    expected_outputs = [
        "counties.timeseries.json",
        "counties.csv",
        # TODO: Add aggregate timeseries csv back in
        # "counties.timeseries.csv",
        "counties.json",
        "county/36061.json",
        "county/36061.timeseries.json",
    ]

    output_paths = [
        str(path.relative_to(tmp_path)) for path in tmp_path.glob("**/*") if not path.is_dir()
    ]
    assert sorted(output_paths) == sorted(expected_outputs)


def test_output_no_timeseries_rows(nyc_region, tmp_path):
    regional_input = api_v2_pipeline.RegionalInput.from_region_and_model_output(
        nyc_region, tmp_path
    )

    # Creating a new regional input with an empty timeseries dataset
    timeseries = regional_input.timeseries
    timeseries_data = timeseries.data.loc[timeseries.data.fips.isna()]
    regional_data = combined_datasets.RegionalData(
        regional_input.region, regional_input.latest, OneRegionTimeseriesDataset(timeseries_data),
    )
    regional_input = api_v2_pipeline.RegionalInput(
        regional_input.region, regional_input.model_output, regional_data,
    )
    assert regional_input.timeseries.empty

    all_timeseries_api = api_v2_pipeline.run_on_regions([regional_input])

    assert all_timeseries_api
