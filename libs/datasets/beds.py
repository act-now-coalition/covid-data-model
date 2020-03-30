from typing import List, Iterator
import enum
import pandas as pd
import datetime
from libs.datasets import data_source
from libs.datasets import AggregationLevel


class BedsDataset(object):
    class Fields(object):
        STATE = "state"
        COUNTY = "county"
        STAFFED_BEDS = "staffed_beds"
        LICENSED_BEDS = "licensed_beds"
        ICU_BEDS = "icu_beds"

        # Name of source of dataset, i.e. JHU
        SOURCE = "source"
        AGGREGATE_LEVEL = "aggregate_level"
        GENERATED = 'generated'

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
            data = cls._fill_missing_state_with_county(data)

        return cls(data)

    @classmethod
    def _fill_missing_state_with_county(cls, data):
        state_groupby_fields = [cls.Fields.SOURCE, cls.Fields.STATE]

        county_data = data[data.aggregate_level == 'county']
        state_data = county_data.groupby(state_groupby_fields).sum().reset_index()
        state_data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.STATE.value
        state_data = state_data.set_index(state_groupby_fields)
        state_data[cls.Fields.GENERATED] = True

        existing_state = data[data.aggregate_level == AggregationLevel.STATE.value]
        existing_state = existing_state.set_index(state_groupby_fields)

        non_matching = state_data[~state_data.index.isin(existing_state.index)]
        return pd.concat([
            data,
            non_matching.reset_index()
        ])
