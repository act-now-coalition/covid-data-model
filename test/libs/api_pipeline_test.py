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


# def test_run_summary
