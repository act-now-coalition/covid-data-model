import pytest
from libs.datasets import dataset_cache


@pytest.fixture(scope="session", autouse=True)
def set_timeseries_dataset_cache():
    dataset_cache.set_pickle_cache_tempdir()
