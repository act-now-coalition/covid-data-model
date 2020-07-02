import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
from libs.datasets import data_source
from libs.datasets import dataset_utils


class TexasHospitalizations(data_source.DataSource):
    DATA_PATH = "data/states/tx/tx_fips_hospitalizations.csv"
    SOURCE_NAME = "tx_hosp"
    FILL_MISSING_STATE_LEVEL_DATA = False

    INDEX_FIELD_MAP = {
        CommonFields.DATE: CommonFields.DATE,
        CommonFields.AGGREGATE_LEVEL: CommonFields.AGGREGATE_LEVEL,
        CommonFields.COUNTRY: CommonFields.COUNTRY,
        CommonFields.STATE: CommonFields.STATE,
        CommonFields.FIPS: CommonFields.FIPS,
    }

    COMMON_FIELD_MAP = {
        CommonFields.CURRENT_HOSPITALIZED: CommonFields.CURRENT_HOSPITALIZED,
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        return cls(data)
