import pathlib
import pytest


@pytest.fixture
def nyc_fips():
    return "36061"


@pytest.fixture
def nyc_model_output_path() -> pathlib.Path:
    # generated from running pyseir model output.  To update, run
    test_root = pathlib.Path(__file__).parent
    return test_root / "data" / "pyseir" / "36061.1.json"
