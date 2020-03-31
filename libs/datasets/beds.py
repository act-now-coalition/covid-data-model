from typing import List, Iterator
import enum
import pandas as pd
import datetime
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel


class BedsDataset(object):
    class Fields(object):
        STATE = "state"
        FIPS = "fips"
        STAFFED_BEDS = "staffed_beds"
        LICENSED_BEDS = "licensed_beds"
        ICU_BEDS = "icu_beds"

        # Name of source of dataset, i.e. JHU
        COUNTY = "county"
        SOURCE = "source"
        AGGREGATE_LEVEL = "aggregate_level"
        GENERATED = "generated"

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_source(cls, source: data_source.DataSource, fill_missing_state=True):
        """Loads data from a specific datasource."""
        if not source.BEDS_FIELD_MAP:
            raise ValueError("Source must have beds field map.")

        data = source.data
        to_common_fields = {value: key for key, value in source.BEDS_FIELD_MAP.items()}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        data[cls.Fields.SOURCE] = source.SOURCE_NAME
        data[cls.Fields.GENERATED] = False

        if fill_missing_state:
            state_groupby_fields = [cls.Fields.SOURCE, cls.Fields.STATE]
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data,
                state_groupby_fields,
                AggregationLevel.COUNTY,
                AggregationLevel.STATE,
            ).reset_index()

            non_matching[cls.Fields.GENERATED] = True
            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)

        return cls(data)

    def get_state_level(self, state):
        aggregation_filter = self.data.aggregate_level == AggregationLevel.STATE.value

        icu_beds = self.data[(self.data.state == state) & aggregation_filter].icu_beds
        if len(icu_beds):
            return icu_beds.iloc[0]

        return None

    def get_county_level(self, state, county):
        aggregation_filter = self.data.aggregate_level == AggregationLevel.COUNTY.value
        state_filter = self.data.state == state
        county_filter = self.data.county == county
        icu_beds = self.data[state_filter & aggregation_filter & county_filter].icu_beds
        if len(icu_beds):
            return icu_beds.iloc[0]

        return None
