import pathlib
import unittest

from libs import pipeline
from libs.datasets import combined_datasets
from libs.pipeline import Region
from pyseir import cli
from pyseir.inference import whitelist
from pyseir.utils import get_run_artifact_path, RunArtifact
import libs.datasets.can_model_output_schema as schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.functions import generate_api
import pytest

# turns all warnings into errors for this module


# Suppressing Matplotlib RuntimeWarning for Figure Gen Count right now. The regex for message isn't
# (https://stackoverflow.com/questions/27476642/matplotlib-get-rid-of-max-open-warning-output)
@pytest.mark.filterwarnings("error", "ignore::RuntimeWarning")
@pytest.mark.slow
def test_pyseir_end_to_end_idaho(tmp_path):
    # This covers a lot of edge cases.
    with unittest.mock.patch("pyseir.utils.OUTPUT_DIR", str(tmp_path)):
        fips = "16001"
        region = Region.from_fips(fips)
        pipelines = cli._build_all_for_states(states=["ID"], fips=fips)
        cli._write_pipeline_output(pipelines, tmp_path)
        path = get_run_artifact_path(region, RunArtifact.WEB_UI_RESULT).replace(
            "__INTERVENTION_IDX__", "2"
        )
        path = pathlib.Path(path)
        assert path.exists()
        output = CANPyseirLocationOutput.load_from_path(path)

        latest = combined_datasets.get_us_latest_for_fips(fips)
        timeseries = combined_datasets.get_timeseries_for_fips(fips)
        summary = generate_api.generate_region_summary(latest, None, output)
        timeseries_output = generate_api.generate_region_timeseries(summary, timeseries, [], output)
        assert timeseries_output

        data = output.data
        with_values = data[schema.RT_INDICATOR].dropna()
        assert len(with_values) > 10
        assert (with_values > 0).all()
        assert (with_values < 6).all()


@pytest.mark.filterwarnings("error")
@pytest.mark.slow
@pytest.mark.parametrize("fips,expected_results", [(None, True), ("16013", True), ("26013", False)])
def test_filters_counties_properly(fips, expected_results):
    whitelist_df = cli._generate_whitelist()
    state_regions = [pipeline.Region.from_state("ID")]
    results = whitelist.regions_in_states(state_regions, whitelist_df, fips=fips)
    if fips and expected_results:
        assert results == [pipeline.Region.from_fips(fips)]
    elif expected_results:
        assert 30 < len(results) <= 44  # Whitelisted ID counties.

    if not expected_results:
        assert results == []
