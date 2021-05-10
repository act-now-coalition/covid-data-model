import pytest
from click.testing import CliRunner

from cli import data


@pytest.mark.skip(reason="mysteriously crashes on server, see PR 969")
@pytest.mark.slow
def test_population_filter(tmp_path):
    runner = CliRunner()
    output_path = tmp_path / "filtered.csv"
    runner.invoke(
        data.run_population_filter, [str(output_path)], catch_exceptions=False,
    )
    assert output_path.exists()
