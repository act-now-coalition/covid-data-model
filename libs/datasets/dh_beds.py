import numpy
import pandas as pd
from libs.datasets.beds import BedsDataset
from libs.datasets import dataset_utils
from libs.datasets import data_source


class DHBeds(data_source.DataSource):
    DATA_PATH = "data/beds-dh/hospital_beds_by_county.csv"
    SOURCE_NAME = "DH"

    class Fields(object):
        STATE = "state"
        COUNTY = "county"
        STAFFED_BEDS = "staffed_beds"
        LICENSED_BEDS = "licensed_beds"
        ICU_BEDS = "icu_beds"

        AGGREGATE_LEVEL = "aggregate_level"

    BEDS_FIELD_MAP = {
        BedsDataset.Fields.STATE: Fields.STATE,
        BedsDataset.Fields.COUNTY: Fields.COUNTY,
        BedsDataset.Fields.STAFFED_BEDS: Fields.STAFFED_BEDS,
        BedsDataset.Fields.LICENSED_BEDS: Fields.LICENSED_BEDS,
        BedsDataset.Fields.ICU_BEDS: Fields.ICU_BEDS,
        BedsDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL
    }

    def __init__(self, path):
        data = pd.read_csv(path)
        self.data = self.standardize_data(data)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        # All DH data is aggregated at the county level
        data[cls.Fields.AGGREGATE_LEVEL] = 'county'
        return data

    @classmethod
    def build_from_local_github(cls) -> "JHUTimeseriesData":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_PATH)
