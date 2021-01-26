from functools import lru_cache

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries


class CDCTestingDataset(data_source.DataSource):
    SOURCE_NAME = "CDCTesting"

    COMMON_DF_CSV_PATH = "data/testing-cdc/timeseries-common.csv"

    EXPECTED_FIELDS = [
        CommonFields.TEST_POSITIVITY_7D,
    ]
