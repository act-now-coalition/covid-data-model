import os
import pathlib
import json
import pandas as pd
from pyseir import cli
from pyseir.utils import get_run_artifact_path, RunArtifact
import libs.datasets.can_model_output_schema as schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
import pytest

# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.slow
def test_pyseir_end_to_end_idaho():
    # This covers a lot of edge cases.
    # cli._run_all(state='Idaho')
    cli._build_all_for_states(states=["Idaho"], fips="16001")
    path = get_run_artifact_path("16001", RunArtifact.WEB_UI_RESULT).replace(
        "__INTERVENTION_IDX__", "2"
    )
    path = pathlib.Path(path)
    assert path.exists()
    output = CANPyseirLocationOutput.load_from_path(path)
    data = output.data
    with_values = data[schema.RT_INDICATOR].dropna()
    assert len(with_values) > 10
    assert (with_values > 0).all()
    assert (with_values < 6).all()


@pytest.mark.slow
@pytest.mark.parametrize("fips,expected_results", [(None, True), ("16013", True), ("26013", False)])
def test_filters_counties_properly(fips, expected_results):
    cli._generate_whitelist()
    results = cli.build_counties_to_run_per_state(["Idaho"], fips=fips)
    if fips and expected_results:
        assert results == {fips: "Idaho"}
    elif expected_results:
        assert results

    if not expected_results:
        assert results == {}
