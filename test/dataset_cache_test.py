from libs.datasets import dataset_cache
import pytest
import pandas as pd
from libs.datasets.latest_values_dataset import LatestValuesDataset


@pytest.fixture(autouse=True)
def clear_cache_keys():
    # Clears existing cache keys so multiple tests can start with a clean slate
    # and create functions that share the same name across tests.
    yield
    dataset_cache._EXISTING_CACHE_KEYS = set()


def test_uses_cache_when_env_variable_saved(monkeypatch, tmp_path):

    data = pd.DataFrame([{"date": "2020-01-01", "state": "DC", "fips": "11"}])
    dataset = LatestValuesDataset(data)
    calls = 0

    original_records = dataset.data.to_dict(orient="records")

    @dataset_cache.cache_dataset_on_disk(LatestValuesDataset)
    def _wrapped_function():
        nonlocal calls
        calls += 1
        return dataset

    monkeypatch.setenv(dataset_cache.PICKLE_CACHE_ENV_KEY, str(tmp_path))

    first_call = _wrapped_function()
    assert first_call.data.to_dict(orient="records") == original_records

    second_call = _wrapped_function()
    assert second_call.data.to_dict(orient="records") == original_records
    # Verify that second call was done through cache
    assert calls == 1


def test_doesnt_cache_without_env(monkeypatch, tmp_path):

    monkeypatch.setenv(dataset_cache.PICKLE_CACHE_ENV_KEY, "")

    data = pd.DataFrame([{"date": "2020-01-01", "state": "DC", "fips": "11"}])
    dataset = LatestValuesDataset(data)
    calls = 0

    @dataset_cache.cache_dataset_on_disk(LatestValuesDataset)
    def _wrapped_function():
        nonlocal calls
        calls += 1
        return dataset

    _wrapped_function()
    _wrapped_function()
    assert calls == 2


def test_wrapping_multiple_times_fails(monkeypatch, tmp_path):

    data = pd.DataFrame([{"date": "2020-01-01", "state": "DC", "fips": "11"}])
    dataset = LatestValuesDataset(data)
    calls = 0

    @dataset_cache.cache_dataset_on_disk(LatestValuesDataset)
    def _wrapped_function():
        nonlocal calls
        calls += 1
        return dataset

    with pytest.raises(ValueError):

        @dataset_cache.cache_dataset_on_disk(LatestValuesDataset)
        def _wrapped_function():
            nonlocal calls
            calls += 1
            return dataset

    @dataset_cache.cache_dataset_on_disk(LatestValuesDataset, key="some_other_name")
    def _wrapped_function():
        nonlocal calls
        calls += 1
        return dataset
