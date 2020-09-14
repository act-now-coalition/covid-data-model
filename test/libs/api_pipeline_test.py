import pytest
import pandas as pd
from libs.enums import Intervention
from libs.pipelines import api_pipeline
from libs.datasets import combined_datasets
from libs.datasets.timeseries import OneRegionTimeseriesDataset


@pytest.mark.slow
@pytest.mark.parametrize(
    "intervention",
    [
        Intervention.OBSERVED_INTERVENTION,
        Intervention.STRONG_INTERVENTION,
        Intervention.NO_INTERVENTION,
    ],
)
def test_build_timeseries_and_summary_outputs(nyc_model_output_path, nyc_region, intervention):

    regional_input = api_pipeline.RegionalInput.from_region_and_intervention(
        nyc_region, intervention, nyc_model_output_path.parent
    )
    timeseries = api_pipeline.build_timeseries_for_region(regional_input)

    if intervention is Intervention.NO_INTERVENTION:
        # Test data does not contain no intervention model, should not output any results.
        assert not timeseries
        return

    assert timeseries

    if intervention is Intervention.STRONG_INTERVENTION:
        assert timeseries.projections
        assert timeseries.timeseries
    elif intervention is Intervention.OBSERVED_INTERVENTION:
        assert not timeseries.projections
        assert not timeseries.timeseries


def test_build_api_output_for_intervention(nyc_region, nyc_model_output_path, tmp_path):
    county_output = tmp_path / "county"
    regional_input = api_pipeline.RegionalInput.from_region_and_intervention(
        nyc_region, Intervention.STRONG_INTERVENTION, nyc_model_output_path.parent
    )
    all_timeseries_api = api_pipeline.run_on_all_regional_inputs_for_intervention([regional_input])

    api_pipeline.deploy_single_level(
        Intervention.STRONG_INTERVENTION, all_timeseries_api, tmp_path, county_output
    )
    expected_outputs = [
        "counties.STRONG_INTERVENTION.timeseries.json",
        "counties.STRONG_INTERVENTION.csv",
        "counties.STRONG_INTERVENTION.timeseries.csv",
        "counties.STRONG_INTERVENTION.json",
        "county/36061.STRONG_INTERVENTION.json",
        "county/36061.STRONG_INTERVENTION.timeseries.json",
    ]

    output_paths = [
        str(path.relative_to(tmp_path)) for path in tmp_path.glob("**/*") if not path.is_dir()
    ]
    assert sorted(output_paths) == sorted(expected_outputs)


def test_output_no_timeseries_rows(nyc_region, tmp_path):
    regional_input = api_pipeline.RegionalInput.from_region_and_intervention(
        nyc_region, Intervention.OBSERVED_INTERVENTION, tmp_path
    )

    # Creating a new regional input with an empty timeseries dataset
    timeseries = regional_input.timeseries
    timeseries_data = timeseries.data.loc[timeseries.data.fips.isna()]
    regional_data = combined_datasets.RegionalData(
        regional_input._combined_data.region,
        regional_input.latest,
        OneRegionTimeseriesDataset(timeseries_data),
    )
    regional_input = api_pipeline.RegionalInput(
        regional_input.region,
        regional_input.model_output,
        regional_input.intervention,
        regional_data,
    )
    assert regional_input.timeseries.empty

    all_timeseries_api = api_pipeline.run_on_all_regional_inputs_for_intervention([regional_input])

    assert all_timeseries_api
