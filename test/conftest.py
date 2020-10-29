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
def rt_dataset():
    test_root = pathlib.Path(__file__).parent
    path = test_root / "data" / "pyseir" / "rt_combined_metric.csv"
    return timeseries.MultiRegionTimeseriesDataset.from_csv(path)


@pytest.fixture
def icu_dataset():
    test_root = pathlib.Path(__file__).parent
    path = test_root / "data" / "pyseir" / "icu_combined_metric.csv"
    return timeseries.MultiRegionTimeseriesDataset.from_csv(path)


@pytest.fixture
def nyc_rt_dataset(nyc_region, rt_dataset) -> timeseries.OneRegionTimeseriesDataset:
    # generated from running pyseir model output.
    return rt_dataset.get_one_region(nyc_region)


@pytest.fixture
def nyc_icu_dataset(nyc_region, icu_dataset) -> timeseries.OneRegionTimeseriesDataset:
    # generated from running pyseir model output.
    return icu_dataset.get_one_region(nyc_region)
