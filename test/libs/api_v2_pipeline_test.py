import pytest
from covidactnow.datapublic.common_fields import CommonFields

from libs import test_positivity
from libs.datasets import timeseries
from libs.pipeline import Region
from libs.pipelines import api_v2_pipeline
from libs.datasets import combined_datasets
from libs.datasets import AggregationLevel
import pandas as pd
import structlog


@pytest.fixture
def nyc_regional_input(nyc_region, rt_dataset, icu_dataset):
    us_dataset = combined_datasets.load_us_timeseries_dataset()
    # Not using test_positivity because currently we don't have any data for counties
    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        nyc_region, us_dataset, rt_dataset, icu_dataset
    )


@pytest.fixture
def il_regional_input(rt_dataset, icu_dataset):
    region = Region.from_state("IL")
    regional_data = combined_datasets.load_us_timeseries_dataset().get_regions_subset([region])
    regional_data = test_positivity.run_and_maybe_join_columns(
        regional_data, structlog.get_logger()
    )

    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        region, regional_data, rt_dataset, icu_dataset
    )


@pytest.fixture
def il_regional_input_empty_test_positivity_column(rt_dataset, icu_dataset):
    region = Region.from_state("IL")
    regional_data = combined_datasets.load_us_timeseries_dataset().get_regions_subset([region])
    empty_test_positivity = timeseries.MultiRegionDataset.from_geodata_timeseries_df(
        # Create with explicit dtypes because to_numeric in from_geodata_timeseries_df doesn't work with
        # empty DataFrames.
        pd.DataFrame(
            {
                CommonFields.LOCATION_ID: pd.Series([], dtype="str"),
                CommonFields.DATE: pd.Series([], dtype="datetime64[ns]"),
                CommonFields.TEST_POSITIVITY: pd.Series([], dtype="float"),
            }
        )
    )

    regional_data = regional_data.drop_column_if_present(CommonFields.TEST_POSITIVITY).join_columns(
        empty_test_positivity
    )
    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        region, regional_data, rt_dataset, icu_dataset
    )


def test_build_timeseries_and_summary_outputs(nyc_regional_input):
    timeseries = api_v2_pipeline.build_timeseries_for_region(nyc_regional_input)
    assert timeseries
    assert timeseries.riskLevels.testPositivityRatio
    assert timeseries.metrics.testPositivityRatioDetails.source


def test_build_timeseries_and_summary_outputs_for_il_state(il_regional_input):
    timeseries = api_v2_pipeline.build_timeseries_for_region(il_regional_input)
    assert timeseries


def test_build_timeseries_and_summary_outputs_for_il_state_with_empty_test_postivity_columnn(
    il_regional_input_empty_test_positivity_column,
):
    timeseries = api_v2_pipeline.build_timeseries_for_region(
        il_regional_input_empty_test_positivity_column
    )
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
    one_region = combined_datasets.load_us_timeseries_dataset().get_one_region(
        nyc_regional_input.region
    )
    regional_input = api_v2_pipeline.RegionalInput(
        nyc_regional_input.region, one_region, None, None
    )
    assert not regional_input.timeseries.empty

    all_timeseries_api = api_v2_pipeline.run_on_regions([regional_input])

    assert all_timeseries_api
