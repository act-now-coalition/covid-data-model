import pathlib
import pytest
from libs.datasets import dataset_cache


@pytest.fixture(scope="session", autouse=True)
def set_timeseries_dataset_cache():
    # Forcing cache to use a new folder to always regenerate cache
    # during tests.
    dataset_cache.set_pickle_cache_dir(force=True, cache_dir=None)


@pytest.fixture
def nyc_fips():
    return "36061"


@pytest.fixture
def nyc_model_output_path() -> pathlib.Path:
    # generated from running pyseir model output.  To update, run
    test_root = pathlib.Path(__file__).parent
    return test_root / "data" / "pyseir" / "36061.1.json"
