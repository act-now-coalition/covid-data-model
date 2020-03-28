import pathlib
import pytest

from click.testing import CliRunner

import run
from libs import build_params


@pytest.mark.parametrize('dataset_name', ['JHU', 'CDS'])
def test_runs_all_interventions_on_one_state(tmpdir, dataset_name):
    runner = CliRunner()

    runner.invoke(run.main, [
        'run-model', dataset_name, '--state', 'MA', '--output-dir', tmpdir
    ], catch_exceptions=False)

    output_path = pathlib.Path(tmpdir)
    output_paths = list(output_path.iterdir())

    assert len(output_paths) == len(build_params.INTERVENTIONS)
    for path in output_paths:
        assert path.read_bytes()


# TODO(chris): Would like to add a test that runs all the states on one intervention.
# I don't think that we should run the entire generation in the unittests.
