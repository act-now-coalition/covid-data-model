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


# @pytest.fixture(scope="module")
# def pyseir_output_path():
#     with tempfile.TemporaryDirectory() as tempdir:
#         cli._build_all_for_states(
#             states=["New York"], generate_reports=False, output_dir=tempdir, fips="36061"
#         )
#         yield fips, pathlib.Path(tempdir)


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


def test_build_api_output_for_intervention(nyc_fips, nyc_model_output_path, tmp_path):
    print(tmp_path)
    us_latest = combined_datasets.build_us_latest_with_all_fields()
    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields()

    nyc_latest = us_latest.get_subset(None, fips=nyc_fips)
    nyc_timeseries = us_timeseries.get_subset(None, fips=nyc_fips)
    all_timeseries_api = api_pipeline.run_on_all_fips_for_intervention(
        nyc_latest, nyc_timeseries, Intervention.STRONG_INTERVENTION, nyc_model_output_path.parent,
    )

    api_pipeline.deploy_single_level(
        Intervention.STRONG_INTERVENTION, all_timeseries_api, tmp_path, tmp_path
    )
    assert 0
