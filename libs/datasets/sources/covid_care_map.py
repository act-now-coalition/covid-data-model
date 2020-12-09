import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets.dataset_utils import LATEST_VALUES_INDEX_FIELDS


class CovidCareMapBeds(data_source.DataSource):
    STATIC_CSV = "data/covid-care-map/static.csv"

    SOURCE_NAME = "CCM"

    INDEX_FIELD_MAP = {f: f for f in LATEST_VALUES_INDEX_FIELDS}

    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.STAFFED_BEDS,
            CommonFields.LICENSED_BEDS,
            CommonFields.ICU_BEDS,
            CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE,
            CommonFields.ICU_TYPICAL_OCCUPANCY_RATE,
            CommonFields.MAX_BED_COUNT,
        }
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.STATIC_CSV
        # Can't use common_df.read_csv because it expects a date column
        data = pd.read_csv(input_path, dtype={CommonFields.FIPS: str})
        return cls(cls._rename_to_common_fields(data))
