import pathlib
import pytest
from libs import pipeline


@pytest.fixture
def nyc_fips():
    return "36061"


@pytest.fixture
def nyc_region(nyc_fips):
    return pipeline.Region.from_fips(nyc_fips)


@pytest.fixture
def nyc_model_output_path() -> pathlib.Path:
    # generated from running pyseir model output.  To update, run
    test_root = pathlib.Path(__file__).parent
    return test_root / "data" / "pyseir" / "36061.1.json"
