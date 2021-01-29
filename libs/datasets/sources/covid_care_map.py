from functools import lru_cache

import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import timeseries


class CovidCareMapBeds(data_source.DataSource):
    STATIC_CSV = "data/covid-care-map/static.csv"

    SOURCE_NAME = "CCM"

    EXPECTED_FIELDS = [
        CommonFields.STAFFED_BEDS,
        CommonFields.LICENSED_BEDS,
        CommonFields.ICU_BEDS,
        CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE,
        CommonFields.ICU_TYPICAL_OCCUPANCY_RATE,
        CommonFields.MAX_BED_COUNT,
    ]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.STATIC_CSV
        # Can't use common_df.read_csv because it expects a date column
        data = pd.read_csv(input_path, dtype={CommonFields.FIPS: str})
        return timeseries.MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)
