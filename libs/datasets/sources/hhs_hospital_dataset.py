from functools import lru_cache
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.timeseries import MultiRegionDataset


class HHSHospitalDataset(data_source.DataSource):
    SOURCE_NAME = "HHSHospital"

    COMMON_DF_CSV_PATH = "data/hospital-hhs/timeseries-common.csv"

    EXPECTED_FIELDS = [
        CommonFields.ICU_BEDS,
        CommonFields.CURRENT_ICU_TOTAL,
        CommonFields.CURRENT_ICU,
        CommonFields.STAFFED_BEDS,
        CommonFields.HOSPITAL_BEDS_IN_USE_ANY,
        CommonFields.CURRENT_HOSPITALIZED,
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> MultiRegionDataset:
        return super().make_dataset().latest_in_static(CommonFields.ICU_BEDS)
