import pytest
from covidactnow.datapublic.common_fields import CommonFields

from libs.enums import Intervention
from libs.pipelines import api_pipeline
from libs.datasets import combined_datasets
from libs.datasets.timeseries import OneRegionTimeseriesDataset
import pandas as pd


@pytest.mark.slow
@pytest.mark.parametrize(
    "intervention",
    [
        Intervention.OBSERVED_INTERVENTION,
        Intervention.STRONG_INTERVENTION,
        Intervention.NO_INTERVENTION,
    ],
)
def test_build_timeseries_and_summary_outputs(nyc_region, intervention, rt_dataset, icu_dataset):

    regional_input = api_pipeline.RegionalInput.from_region_and_intervention(
        nyc_region, intervention, rt_dataset, icu_dataset
    )
    timeseries = api_pipeline.build_timeseries_for_region(regional_input)

    assert timeseries
    assert not timeseries.projections
    assert not timeseries.timeseries


def test_build_api_output_for_intervention(nyc_region, tmp_path, rt_dataset, icu_dataset):
    county_output = tmp_path / "county"
    regional_input = api_pipeline.RegionalInput.from_region_and_intervention(
        nyc_region, Intervention.STRONG_INTERVENTION, rt_dataset, icu_dataset,
    )
    all_timeseries_api = api_pipeline.run_on_all_regional_inputs_for_intervention([regional_input])

    api_pipeline.deploy_single_level(
        Intervention.STRONG_INTERVENTION, all_timeseries_api, tmp_path, county_output
    )
    expected_outputs = [
        "counties.STRONG_INTERVENTION.timeseries.json",
        "counties.STRONG_INTERVENTION.csv",
        # No projections are being generated so
        # "counties.STRONG_INTERVENTION.timeseries.csv",
        "counties.STRONG_INTERVENTION.json",
        "county/36061.STRONG_INTERVENTION.json",
        "county/36061.STRONG_INTERVENTION.timeseries.json",
    ]

    output_paths = [
        str(path.relative_to(tmp_path)) for path in tmp_path.glob("**/*") if not path.is_dir()
    ]

    assert sorted(output_paths) == sorted(expected_outputs)


def test_output_no_timeseries_rows(nyc_region, rt_dataset, icu_dataset):
    regional_input = api_pipeline.RegionalInput.from_region_and_intervention(
        nyc_region, Intervention.OBSERVED_INTERVENTION, rt_dataset, icu_dataset
    )

    # Creating a new regional input with an empty timeseries dataset
    timeseries_data = pd.DataFrame([], columns=[CommonFields.LOCATION_ID, CommonFields.DATE])
    regional_data = combined_datasets.RegionalData(
        regional_input.region, OneRegionTimeseriesDataset(timeseries_data, regional_input.latest),
    )
    regional_input = api_pipeline.RegionalInput(
        regional_input.region, None, None, regional_input.intervention, regional_data,
    )
    assert regional_input.timeseries.empty

    all_timeseries_api = api_pipeline.run_on_all_regional_inputs_for_intervention([regional_input])

    assert all_timeseries_api
