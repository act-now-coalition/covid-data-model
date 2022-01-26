from functools import lru_cache

import pandas as pd
from datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import timeseries


class CANLocationPageURLS(data_source.DataSource):
    SOURCE_TYPE = "can_urls"

    STATIC_CSV = "data/misc/can_location_page_urls.csv"

    EXPECTED_FIELDS = [
        CommonFields.CAN_LOCATION_PAGE_URL,
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.STATIC_CSV
        # Can't use common_df.read_csv because it expects a date column
        data = pd.read_csv(input_path, dtype={CommonFields.FIPS: str})
        return timeseries.MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)
