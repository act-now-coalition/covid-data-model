
import pytest

from libs.datasets import combined_datasets

from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import JHUDataset
from libs.datasets import CDSDataset
from libs.datasets import CovidTrackingDataSource
from libs.datasets import NevadaHospitalAssociationData


# Tests to make sure that combined datasets are building data with unique indexes
# If this test is failing, it means that there is one of the data sources that
# is returning multiple values for a single row.
def test_unique_index_values_us_timeseries():
    timeseries = combined_datasets.build_us_timeseries_with_all_fields()
    timeseries_data = timeseries.data.set_index(timeseries.INDEX_FIELDS)
    duplicates = timeseries_data.index.duplicated()
    assert not sum(duplicates)


def test_unique_index_values_us_latest():
    latest = combined_datasets.build_us_latest_with_all_fields()
    latest_data = latest.data.set_index(latest.INDEX_FIELDS)
    duplicates = latest_data.index.duplicated()
    assert not sum(duplicates)


@pytest.mark.parametrize("data_source_cls", [
    JHUDataset,
    CDSDataset,
    CovidTrackingDataSource,
    NevadaHospitalAssociationData,

])
def test_unique_timeseries(data_source_cls):

    data_source = data_source_cls.local()
    timeseries = TimeseriesDataset.build_from_data_source(data_source)
    timeseries = combined_datasets.US_STATES_FILTER.apply(timeseries)
    timeseries_data = timeseries.data.set_index(timeseries.INDEX_FIELDS)
    duplicates = timeseries_data.index.duplicated()
    assert not sum(duplicates)
