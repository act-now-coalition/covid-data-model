from functools import lru_cache

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


class HHSTestingDataset(data_source.DataSource):
    SOURCE_NAME = "HHSTesting"

    DATA_PATH = "data/testing-hhs/timeseries-common.csv"

    COMMON_FIELD_MAP = {f: f for f in {CommonFields.NEGATIVE_TESTS, CommonFields.POSITIVE_TESTS,}}

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        return cls.make_timeseries_dataset(data)
