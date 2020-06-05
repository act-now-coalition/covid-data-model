import pandas as pd
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.common_fields import CommonFields


class CmdcDataSource(data_source.DataSource):
    DATA_PATH = "data/cases-cmdc/timeseries-common.csv"
    SOURCE_NAME = "CMDC"

    INDEX_FIELD_MAP = {
        CommonFields.DATE: CommonFields.DATE,
        CommonFields.AGGREGATE_LEVEL: CommonFields.AGGREGATE_LEVEL,
        CommonFields.COUNTRY: CommonFields.COUNTRY,
        CommonFields.STATE: CommonFields.STATE,
        CommonFields.FIPS: CommonFields.FIPS,
    }

    # Keep in sync with update_cmdc.py in the covid-data-public repo.
    COMMON_FIELD_MAP = {
        CommonFields.DEATHS: CommonFields.DEATHS,
        CommonFields.CURRENT_ICU_TOTAL: CommonFields.CURRENT_ICU_TOTAL,
        CommonFields.NEGATIVE_TESTS: CommonFields.NEGATIVE_TESTS,
        CommonFields.POSITIVE_TESTS: CommonFields.POSITIVE_TESTS,
        CommonFields.CURRENT_VENTILATED: CommonFields.CURRENT_VENTILATED,
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(
            input_path, parse_dates=[CommonFields.DATE], dtype={CommonFields.FIPS: str}
        )
        return cls(data)
