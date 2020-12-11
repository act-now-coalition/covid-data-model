from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


class HHSHospitalDataset(data_source.DataSource):
    SOURCE_NAME = "HHSHospital"

    DATA_PATH = "data/hospital-hhs/timeseries-common.csv"

    INDEX_FIELD_MAP = {f: f for f in TIMESERIES_INDEX_FIELDS}

    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.ICU_BEDS,
            CommonFields.CURRENT_ICU_TOTAL,
            CommonFields.CURRENT_ICU,
            CommonFields.STAFFED_BEDS,
            CommonFields.CURRENT_HOSPITALIZED_TOTAL,
            CommonFields.CURRENT_HOSPITALIZED,
        }
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path, set_index=False)
        return cls(data)
