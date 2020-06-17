import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils


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
    # DataSource objects must have a map from CommonFields to fields in the source file. For CMDC the
    # conversion is done in the covid-data-public repo so the map here doesn't represent any field renaming.
    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.CASES,
            CommonFields.DEATHS,
            CommonFields.CURRENT_ICU,
            CommonFields.CURRENT_ICU_TOTAL,
            CommonFields.NEGATIVE_TESTS,
            CommonFields.POSITIVE_TESTS,
            CommonFields.STAFFED_BEDS,
            CommonFields.HOSPITAL_BEDS_IN_USE_ANY,
            CommonFields.CURRENT_VENTILATED,
            CommonFields.CURRENT_HOSPITALIZED,
            CommonFields.ICU_BEDS,
        }
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(
            input_path, parse_dates=[CommonFields.DATE], dtype={CommonFields.FIPS: str}
        )
        return cls(data)
