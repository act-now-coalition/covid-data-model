import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets.dataset_utils import STATIC_INDEX_FIELDS


class CANLocationPageURLS(data_source.DataSource):
    STATIC_CSV = "data/misc/can_location_page_urls.csv"

    SOURCE_NAME = "can_urls"

    INDEX_FIELD_MAP = {f: f for f in STATIC_INDEX_FIELDS}

    COMMON_FIELD_MAP = {f: f for f in {CommonFields.CAN_LOCATION_PAGE_URL,}}

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.STATIC_CSV
        # Can't use common_df.read_csv because it expects a date column
        data = pd.read_csv(input_path, dtype={CommonFields.FIPS: str})
        return cls(cls._rename_to_common_fields(data))
