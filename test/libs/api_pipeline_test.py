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

    summary, timeseries = api_pipeline.build_summary_and_timeseries_for_fips(
        nyc_fips, intervention, us_latest, us_timeseries, nyc_model_output_path.parent
    )

    if intervention is Intervention.NO_INTERVENTION:
        # Test data does not contain no intervention model, should not output any results.
        assert not summary
        assert not timeseries
        return

    assert summary
    assert timeseries

    if intervention is Intervention.STRONG_INTERVENTION:
        assert summary.projections
        assert timeseries.projections
        assert timeseries.timeseries
    elif intervention is Intervention.OBSERVED_INTERVENTION:
        assert not summary.projections
        assert not timeseries.projections
        assert not timeseries.timeseries
