import os
import unittest

from libs.pipeline import Region
from pyseir.utils import RunArtifact
from pyseir.utils import get_run_artifact_path


def test(tmp_path):
    with unittest.mock.patch("pyseir.utils.OUTPUT_DIR", str(tmp_path)):

        path = get_run_artifact_path(Region.from_state("TX"), RunArtifact.RT_INFERENCE_REPORT)
        assert path == os.path.join(
            tmp_path, "pyseir/state_summaries/reports/Rt_results__Texas__48.pdf"
        )

        path = get_run_artifact_path(Region.from_fips("48301"), RunArtifact.RT_SMOOTHING_REPORT)
        assert path == os.path.join(
            tmp_path, "pyseir/Texas/reports/Rt_smoothing__Texas__Loving County__48301.pdf"
        )

        path = get_run_artifact_path(
            Region.from_cbsa_code("10100"), RunArtifact.RT_INFERENCE_REPORT
        )
        assert path == os.path.join(
            tmp_path, "pyseir/state_summaries/reports/Rt_results__CBSA__10100.pdf"
        )
