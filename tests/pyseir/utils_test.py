import os
import unittest

from libs.pipeline import Region
from pyseir import OUTPUT_DIR
from pyseir.utils import RunArtifact
from pyseir.utils import get_run_artifact_path


def test():
    output_dir = "/tmp/test/output"
    with unittest.mock.patch("pyseir.utils.OUTPUT_DIR", output_dir):
        assert get_run_artifact_path(
            Region.from_state("TX"), RunArtifact.RT_INFERENCE_REPORT
        ) == os.path.join(output_dir, "pyseir/state_summaries/reports/Rt_results__Texas__48.pdf")
        assert get_run_artifact_path(
            Region.from_fips("48301"), RunArtifact.RT_SMOOTHING_REPORT
        ) == os.path.join(
            output_dir, "pyseir/Texas/reports/Rt_smoothing__Texas__Loving County__48301" ".pdf"
        )
        assert get_run_artifact_path(
            Region.from_cbsa_code("10100"), RunArtifact.RT_INFERENCE_REPORT
        ) == os.path.join(output_dir, "pyseir/state_summaries/reports/Rt_results__CBSA__10100.pdf")
