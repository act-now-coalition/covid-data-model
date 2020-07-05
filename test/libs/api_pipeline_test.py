import datetime
import pathlib
import tempfile
import pytest
from libs.functions import generate_api
from libs.pipelines import api_pipeline
from libs.datasets import combined_datasets
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.enums import Intervention
from api.can_api_definition import CovidActNowAreaSummary
from api.can_api_definition import _Actuals
from api.can_api_definition import _Projections
from api.can_api_definition import _ResourceUsageProjection

from pyseir import cli

NYC_FIPS = "36061"


@pytest.mark.parametrize(
    "intervention",
    [
        Intervention.OBSERVED_INTERVENTION,
        Intervention.STRONG_INTERVENTION,
        Intervention.NO_INTERVENTION,
    ],
)
def test_build_timeseries_and_summary_outputs(nyc_model_output_path, nyc_fips, intervention):

    us_latest = combined_datasets.build_us_latest_with_all_fields()
    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields()

    timeseries = api_pipeline.build_timeseries_for_fips(
        intervention, us_latest, us_timeseries, nyc_model_output_path.parent, nyc_fips
    )

    # TODO(chris): Uncomment and replace when API is outputting for all counties.
    # if intervention is Intervention.NO_INTERVENTION:
    if intervention is not Intervention.STRONG_INTERVENTION:
        # Test data does not contain no intervention model, should not output any results.
        assert not timeseries
        return

    assert timeseries

    if intervention is Intervention.STRONG_INTERVENTION:
        assert timeseries.projections
        assert timeseries.timeseries
    # TODO(chris): Uncomment when API is outputting for all counties
    # elif intervention is Intervention.OBSERVED_INTERVENTION:
    #     assert not timeseries.projections
    #     assert not timeseries.timeseries


def test_build_api_output_for_intervention(nyc_fips, nyc_model_output_path, tmp_path):
    county_output = tmp_path / "county"
    us_latest = combined_datasets.build_us_latest_with_all_fields()
    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields()

    nyc_latest = us_latest.get_subset(None, fips=nyc_fips)
    nyc_timeseries = us_timeseries.get_subset(None, fips=nyc_fips)
    all_timeseries_api = api_pipeline.run_on_all_fips_for_intervention(
        nyc_latest, nyc_timeseries, Intervention.STRONG_INTERVENTION, nyc_model_output_path.parent,
    )

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
