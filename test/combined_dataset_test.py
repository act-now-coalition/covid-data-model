import logging
import re

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


@pytest.mark.parametrize(
    "data_source_cls",
    [JHUDataset, CDSDataset, CovidTrackingDataSource, NevadaHospitalAssociationData,],
)
def test_unique_timeseries(data_source_cls):
    data_source = data_source_cls.local()
    timeseries = TimeseriesDataset.build_from_data_source(data_source)
    timeseries = combined_datasets.US_STATES_FILTER.apply(timeseries)
    timeseries_data = timeseries.data.set_index(timeseries.INDEX_FIELDS)
    duplicates = timeseries_data.index.duplicated(keep=False)
    assert not sum(duplicates), str(timeseries_data.loc[duplicates])


@pytest.mark.parametrize(
    "data_source_cls",
    [JHUDataset, CDSDataset, CovidTrackingDataSource, NevadaHospitalAssociationData, ],
)
def test_expected_field_in_sources(data_source_cls):
    data_source = data_source_cls.local()
    # Extract the USA data from the raw DF. Replace this with cleaner access when the DataSource makes it easy.
    rename_columns = {source: common for common, source in data_source.all_fields_map().items()}
    renamed_data = data_source.data.rename(columns=rename_columns)
    usa_data = renamed_data.loc[renamed_data["country"] == "USA"]

    assert not usa_data.empty

    states = set(usa_data["state"])

    if data_source.SOURCE_NAME == "NHA":
        assert states == {"NV"}
    else:
        good_state = set()
        for state in states:
            if re.fullmatch(r'[A-Z]{2}', state):
                good_state.add(state)
            else:
                logging.info(f"Ignoring {state} in {data_source.SOURCE_NAME}")
        assert len(good_state) >= 48
