import pytest
from click.testing import CliRunner


from cli import data


@pytest.mark.slow
def test_summary_save(tmp_path):

    runner = CliRunner()

    filename = "summary.csv"
    runner.invoke(
        data.save_summary,
        ["--output-dir", str(tmp_path), "--level", "state", "--filename", filename],
    )

    output_path = tmp_path / filename
    assert output_path.exists()
