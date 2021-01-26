from functools import lru_cache

import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import AggregationLevel


class TestAndTraceData(data_source.DataSource):
    DATA_PATH = "data/test-and-trace/state_data.csv"
    SOURCE_NAME = "TestAndTrace"

    EXPECTED_FIELDS = [
        CommonFields.CONTACT_TRACERS_COUNT,
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(
            input_path, parse_dates=[CommonFields.DATE], dtype={CommonFields.FIPS: str}
        )
        data[CommonFields.COUNTRY] = "USA"
        data[CommonFields.AGGREGATE_LEVEL] = AggregationLevel.STATE.value
        return cls.make_timeseries_dataset(data)
