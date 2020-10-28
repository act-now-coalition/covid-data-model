import unittest

from libs import pipeline
from libs.pipeline import Region
from pyseir import cli
from pyseir.utils import SummaryArtifact
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
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

        icu_data_path = tmp_path / SummaryArtifact.ICU_METRIC_COMBINED.value
        icu_data = MultiRegionTimeseriesDataset.from_csv(icu_data_path)
        assert icu_data.get_one_region(region)

        rt_data_path = tmp_path / SummaryArtifact.RT_METRIC_COMBINED.value
        rt_data = MultiRegionTimeseriesDataset.from_csv(rt_data_path)
        assert rt_data.get_one_region(region)


@pytest.mark.filterwarnings("error", "ignore::RuntimeWarning")
@pytest.mark.slow
def test_pyseir_end_to_end_dc(tmp_path):
    # Runs over a single state which tests state filtering + running over more than
    # a single fips.
    with unittest.mock.patch("pyseir.utils.OUTPUT_DIR", str(tmp_path)):
        region = Region.from_state("DC")
        pipelines = cli._build_all_for_states(states=["DC"])
        # Checking to make sure that build all for states properly filters and only
        # returns DC data
        assert len(pipelines) == 2
