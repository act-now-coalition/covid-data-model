from functools import lru_cache

import pandas as pd
import json
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import VaccineEligibility
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import timeseries

import pydantic


class CANVaccineEligibility(data_source.DataSource):
    SOURCE_TYPE = "can_urls"

    STATIC_JSON = "data/can_vaccine_eligibility/can_vaccine_eligibility.json"

    EXPECTED_FIELDS = [
        CommonFields.CAN_LOCATION_PAGE_URL,
    ]

    @classmethod
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.STATIC_JSON
        data = json.load(input_path.open())
        data = [
            {
                CommonFields.FIPS: record[CommonFields.FIPS],
                CommonFields.VACCINE_ELIGIBILITY_DATA: VaccineEligibility.parse_obj(record).json(),
            }
            for record in data
        ]

        data = pd.DataFrame(data)
        # Can't use common_df.read_csv because it expects a date column

        return timeseries.MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)
