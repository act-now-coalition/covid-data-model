import unittest

import pyseir.cli
from libs import parallel_utils
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets

from libs.pipeline import Region
from pyseir.cli import RegionPipeline
import pyseir.utils
from pyseir.cli import RegionPipelineInput
from pyseir.cli import _cache_global_datasets
from pyseir.cli import _patch_nola_infection_rate_in_pipelines
from pyseir.cli import root
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

        # prepare data
        one_region_input = combined_datasets.load_us_timeseries_dataset().get_one_region(region)
        region_pipelines = [RegionPipeline.run(one_region_input)]
        region_pipelines = _patch_nola_infection_rate_in_pipelines(region_pipelines)

        model_output = pyseir.cli.PyseirOutputDatasets.from_pipeline_output(region_pipelines)

        assert model_output.icu.get_one_region(region)
        assert model_output.infection_rate.get_one_region(region)


@pytest.mark.filterwarnings("error", "ignore::RuntimeWarning")
@pytest.mark.slow
def test_pyseir_end_to_end_dc(tmp_path):
    # Runs over a single state which tests state filtering + running over more than
    # a single fips.
    with unittest.mock.patch("pyseir.utils.OUTPUT_DIR", str(tmp_path)):
        region = Region.from_state("DC")

        one_region_input = combined_datasets.load_us_timeseries_dataset().get_one_region(region)
        region_pipelines = [RegionPipeline.run(one_region_input)]
        region_pipelines = _patch_nola_infection_rate_in_pipelines(region_pipelines)
        # Checking to make sure that build all for states properly filters and only
        # returns DC data
        assert len(region_pipelines) == 2
