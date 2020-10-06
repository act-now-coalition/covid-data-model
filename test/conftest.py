import pathlib
import pytest
from libs import pipeline
from libs.datasets import timeseries


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


@pytest.fixture
def nyc_rt_dataset(nyc_region) -> timeseries.OneRegionTimeseriesDataset:
    # generated from running pyseir model output.
    test_root = pathlib.Path(__file__).parent
    path = test_root / "data" / "pyseir" / "rt_combined_metric.csv"
    dataset = timeseries.MultiRegionTimeseriesDataset.from_csv(path)
    return dataset.get_one_region(nyc_region)


@pytest.fixture
def nyc_icu_dataset(nyc_region) -> timeseries.OneRegionTimeseriesDataset:
    # generated from running pyseir model output.
    test_root = pathlib.Path(__file__).parent
    path = test_root / "data" / "pyseir" / "icu_combined_metric.csv"
    dataset = timeseries.MultiRegionTimeseriesDataset.from_csv(path)
    return dataset.get_one_region(nyc_region)
