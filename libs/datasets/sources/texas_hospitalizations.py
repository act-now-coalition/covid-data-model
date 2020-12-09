from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


class TexasHospitalizations(data_source.DataSource):
    DATA_PATH = "data/states/tx/tx_fips_hospitalizations.csv"
    SOURCE_NAME = "tx_hosp"

    INDEX_FIELD_MAP = {f: f for f in TIMESERIES_INDEX_FIELDS}

    COMMON_FIELD_MAP = {
        CommonFields.CURRENT_HOSPITALIZED: CommonFields.CURRENT_HOSPITALIZED,
        CommonFields.CURRENT_ICU: CommonFields.CURRENT_ICU,
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        # Column names are already CommonFields so don't need to rename
        return cls(data)
