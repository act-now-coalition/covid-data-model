import pytest
from libs.datasets import dataset_cache


@pytest.fixture(scope="session", autouse=True)
def set_timeseries_dataset_cache():
    # Forcing cache to use a new folder to always regenerate cache
    # during tests.
    dataset_cache.set_pickle_cache_dir(force=True, cache_dir=None)
