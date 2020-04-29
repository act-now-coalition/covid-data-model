import logging
from libs import enums
import pandas as pd
from libs.datasets.beds import BedsDataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import dataset_utils
from libs.datasets import data_source

_logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)

class CovidCareMapBeds(data_source.DataSource):
    COUNTY_DATA_PATH = "data/covid-care-map/healthcare_capacity_data_county.csv"
    STATE_DATA_PATH = "data/covid-care-map/healthcare_capacity_data_state.csv"

    SOURCE_NAME = "CCM"

    class Fields(object):
        FIPS = "fips_code"
        STATE = "State"
        COUNTY = "County Name"
        STAFFED_ALL_BEDS = "Staffed All Beds"
        STAFFED_ICU_BEDS = "Staffed ICU Beds"
        LICENSED_ALL_BEDS = "Licensed All Beds"
        ALL_BED_TYPICAL_OCCUPANCY_RATE = "All Bed Occupancy Rate"
        ICU_TYPICAL_OCCUPANCY_RATE = "ICU Bed Occupancy Rate"

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
        BedsDataset.Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE: Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE,
        BedsDataset.Fields.ICU_TYPICAL_OCCUPANCY_RATE: Fields.ICU_TYPICAL_OCCUPANCY_RATE,
    }

    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame, aggregate_level: AggregationLevel) -> pd.DataFrame:
        # All DH data is aggregated at the county level
        data[cls.Fields.AGGREGATE_LEVEL] = aggregate_level.value
        data[cls.Fields.COUNTRY] = "USA"
        if cls.Fields.FIPS not in data.columns:
            data[cls.Fields.FIPS] = None

        # Override Washoe County ICU capacity with actual numbers.
        print(data[data[cls.Fields.FIPS] == "32031"].head(100))
        data.loc[data[cls.Fields.FIPS] == "32031", [cls.Fields.STAFFED_ICU_BEDS]] = 162
        data.loc[data[cls.Fields.FIPS] == "32031", [cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE]] = 0.35
        print(data[data[cls.Fields.FIPS] == "32031"].head(100))

        # The virgin islands do not currently have associated fips codes.
        # if VI is supported in the future, this should be removed.
        is_virgin_islands = data[cls.Fields.STATE] == 'VI'
        return data[~is_virgin_islands]

    @classmethod
    def local(cls) -> "CovidCareMapBeds":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        # Load county_data
        path = data_root / cls.COUNTY_DATA_PATH
        data = pd.read_csv(path, dtype={cls.Fields.FIPS: str})
        county_data = cls.standardize_data(data, AggregationLevel.COUNTY)

        # Load
        path = data_root / cls.STATE_DATA_PATH
        data = pd.read_csv(path)
        state_data = cls.standardize_data(data, AggregationLevel.STATE)
        return cls(pd.concat([county_data, state_data]))
