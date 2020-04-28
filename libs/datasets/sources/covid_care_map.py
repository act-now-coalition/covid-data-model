import logging
from libs import enums
import pandas as pd
from libs.datasets.beds import BedsDataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import dataset_utils
from libs.datasets import data_source

_logger = logging.getLogger(__name__)


class CovidCareMapBeds(data_source.DataSource):
    DATA_PATH = "data/covid-care-map/healthcare_capacity_data_county.csv"

    SOURCE_NAME = "CCM"

    class Fields(object):
        FIPS = "fips_code"
        STATE = "State"
        COUNTY = "County Name"
        STAFFED_ALL_BEDS = "Staffed All Beds"
        STAFFED_ICU_BEDS = "Staffed ICU Beds"
        LICENSED_ALL_BEDS = "Licensed All Beds"

        # Added in standardize data.
        AGGREGATE_LEVEL = "aggregate_level"
        COUNTRY = "country"

    BEDS_FIELD_MAP = {
        BedsDataset.Fields.COUNTRY: Fields.COUNTRY,
        BedsDataset.Fields.STATE: Fields.STATE,
        BedsDataset.Fields.FIPS: Fields.FIPS,
        BedsDataset.Fields.STAFFED_BEDS: Fields.STAFFED_ALL_BEDS,
        BedsDataset.Fields.LICENSED_BEDS: Fields.LICENSED_ALL_BEDS,
        BedsDataset.Fields.ICU_BEDS: Fields.STAFFED_ICU_BEDS,
        BedsDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    def __init__(self, data):
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        # All DH data is aggregated at the county level
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.COUNTY.value
        data[cls.Fields.COUNTRY] = "USA"

        # The virgin islands do not currently have associated fips codes.
        # if VI is supported in the future, this should be removed.
        is_virgin_islands = data[cls.Fields.STATE] == 'VI'
        return data[~is_virgin_islands]

    @classmethod
    def local(cls) -> "CovidCareMapBeds":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        path = data_root / cls.DATA_PATH
        data = pd.read_csv(path, dtype={cls.Fields.FIPS: str})
        return cls(data)
