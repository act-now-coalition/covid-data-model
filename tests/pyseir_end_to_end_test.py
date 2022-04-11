import unittest
from typing import List

import pyseir.cli
from libs import parallel_utils
from libs.datasets import combined_datasets

from libs.pipeline import Region
from pyseir.run import OneRegionPipeline
import pyseir.utils
from pyseir.cli import _patch_nola_infection_rate_in_pipelines
import pytest

# turns all warnings into errors for this module
# Suppressing Matplotlib RuntimeWarning for Figure Gen Count right now. The regex for message isn't
# (https://stackoverflow.com/questions/27476642/matplotlib-get-rid-of-max-open-warning-output)
@pytest.mark.filterwarnings("error", "ignore::RuntimeWarning")
@pytest.mark.slow
@pytest.mark.skip(reason="Github action runner runs OOM when loading full dataset")
def test_pyseir_end_to_end_california(tmp_path):
    # This covers a lot of edge cases.
    with unittest.mock.patch("pyseir.utils.OUTPUT_DIR", str(tmp_path)):
        fips = "06037"
        region = Region.from_fips(fips)

        # prepare data
        one_region_input = combined_datasets.load_us_timeseries_dataset().get_one_region(region)
        region_pipelines = [OneRegionPipeline.run(one_region_input)]
        region_pipelines = _patch_nola_infection_rate_in_pipelines(region_pipelines)

        model_output = pyseir.run.PyseirOutputDatasets.from_pipeline_output(region_pipelines)

        assert model_output.infection_rate.get_one_region(region)


@pytest.mark.filterwarnings("error", "ignore::RuntimeWarning")
@pytest.mark.slow
@pytest.mark.skip(reason="Github action runner runs OOM when loading full dataset")
def test_pyseir_end_to_end_dc(tmp_path):
    # Runs over a single state which tests state filtering + running over more than
    # a single fips.
    with unittest.mock.patch("pyseir.utils.OUTPUT_DIR", str(tmp_path)):
        regions_dataset = combined_datasets.load_us_timeseries_dataset().get_subset(state="DC")
        regions = [one_region for _, one_region in regions_dataset.iter_one_regions()]
        region_pipelines: List[OneRegionPipeline] = list(
            parallel_utils.parallel_map(OneRegionPipeline.run, regions)
        )
        # Checking to make sure that build all for states properly filters and only
        # returns DC data
        assert len(region_pipelines) == 2

        model_output = pyseir.run.PyseirOutputDatasets.from_pipeline_output(region_pipelines)
        # TODO(tom): Work out why these have only one region where there are two regions in the
        #  input
        assert len([model_output.infection_rate.iter_one_regions()]) == 1
