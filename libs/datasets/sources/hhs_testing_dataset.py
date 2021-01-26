from functools import lru_cache

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


class HHSTestingDataset(data_source.DataSource):
    SOURCE_NAME = "HHSTesting"

    COMMON_DF_CSV_PATH = "data/testing-hhs/timeseries-common.csv"

    EXPECTED_FIELDS = [CommonFields.NEGATIVE_TESTS, CommonFields.POSITIVE_TESTS]
