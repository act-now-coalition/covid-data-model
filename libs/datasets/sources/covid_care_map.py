import logging
from libs import enums
import pandas as pd
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields
from libs.datasets.dataset_utils import AggregationLevel

_logger = logging.getLogger(__name__)


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
        MAX_BED_COUNT = "max_bed_count"

    INDEX_FIELD_MAP = {
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.STAFFED_BEDS: Fields.STAFFED_ALL_BEDS,
        CommonFields.LICENSED_BEDS: Fields.LICENSED_ALL_BEDS,
        CommonFields.ICU_BEDS: Fields.STAFFED_ICU_BEDS,
        CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: Fields.ALL_BED_TYPICAL_OCCUPANCY_RATE,
        CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: Fields.ICU_TYPICAL_OCCUPANCY_RATE,
        CommonFields.MAX_BED_COUNT: Fields.MAX_BED_COUNT,
    }

    def __init__(self, data):
        super().__init__(data)

    @classmethod
    def standardize_data(
        cls, data: pd.DataFrame, aggregate_level: AggregationLevel
    ) -> pd.DataFrame:
        # All DH data is aggregated at the county level
        data[cls.Fields.AGGREGATE_LEVEL] = aggregate_level.value
        data[cls.Fields.COUNTRY] = "USA"
        if cls.Fields.FIPS not in data.columns:
            data[cls.Fields.FIPS] = None

        if aggregate_level == AggregationLevel.COUNTY:
            # Override Washoe County ICU capacity with actual numbers.
            data.loc[data[cls.Fields.FIPS] == "32031", [cls.Fields.STAFFED_ICU_BEDS]] = 162
            # data.loc[data[cls.Fields.FIPS] == "32031", [cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE]] = 0.35
        if aggregate_level == AggregationLevel.STATE:
            # Overriding NV ICU capacity numbers with actuals
            data.loc[data[cls.Fields.STATE] == "NV", [cls.Fields.STAFFED_ICU_BEDS]] = 844
            # occupancy calculated by 4/28 NHA data
            # (icu_beds - (total_icu_beds_used - covid_icu_beds_used)) / icu_beds
            # (844 - (583 - 158)) / 844 == 0.4964
            # data.loc[data[cls.Fields.STATE] == 'NV', [cls.Fields.ICU_TYPICAL_OCCUPANCY_RATE]] = 0.4964

        data[cls.Fields.MAX_BED_COUNT] = data[
            [cls.Fields.STAFFED_ALL_BEDS, cls.Fields.LICENSED_ALL_BEDS]
        ].max(axis=1)

        # The virgin islands do not currently have associated fips codes.
        # if VI is supported in the future, this should be removed.
        is_virgin_islands = data[cls.Fields.STATE] == "VI"
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
