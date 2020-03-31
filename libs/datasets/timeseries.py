from typing import List, Iterator
import enum
import pandas as pd
import datetime
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import AggregationLevel


class TimeseriesDataset(object):
    class Fields(object):
        DATE = "date"
        COUNTRY = "country"
        STATE = "state"
        CASES = "cases"
        DEATHS = "deaths"
        RECOVERED = "recovered"
        FIPS = "fips"

        # Generated
        COUNTY = "county"
        SOURCE = "source"
        IS_SYNTHETIC = "is_synthetic"
        AGGREGATE_LEVEL = "aggregate_level"
        GENERATED = "generated"

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def states(self) -> List:
        return self.data[self.Fields.STATE].dropna().unique().tolist()

    def county_keys(self) -> List:
        # Check to make sure all values are county values
        county_values = (
            self.data[self.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        )
        county_data = self.data[county_values]

        data = county_data.set_index(
            [self.Fields.COUNTRY, self.Fields.STATE, self.Fields.COUNTY]
        )

        return set(data.index.to_list())

    def get_subset(
        self,
        aggregation_level,
        on=None,
        after=None,
        country=None,
        state=None,
        county=None,
        fips=None,
    ) -> "TimeseriesDataset":
        data = self.data

        if aggregation_level:
            data = data[data.aggregate_level == aggregation_level.value]
        if country:
            data = data[data.country == country]
        if state:
            data = data[data.state == state]
        if county:
            data = data[data.county == county]
        if fips:
            data = data[data.fips == fips]

        if on:
            data = data[data.date == on]
        if after:
            data = data[data.date > after]

        return self.__class__(data)

    def get_data(
        self, country=None, state=None, county=None, fips=None
    ) -> pd.DataFrame:
        data = self.data
        if country:
            data = data[data.country == country]
        if state:
            data = data[data.state == state]
        if county:
            data = data[data.county == county]
        if fips:
            data = data[data.fips == fips]
        return data

    @classmethod
    def from_source(cls, source: data_source.DataSource, fill_missing_state=True):
        """Loads data from a specific datasource."""
        if not source.TIMESERIES_FIELD_MAP:
            raise ValueError("Source must have field timeseries field map.")

        data = source.data
        to_common_fields = {
            value: key for key, value in source.TIMESERIES_FIELD_MAP.items()
        }
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        data[cls.Fields.SOURCE] = source.SOURCE_NAME
        data[cls.Fields.GENERATED] = False
        if fill_missing_state:
            data = cls._fill_missing_state_with_county(data)

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)

        # Choosing to sort by date
        data = data.sort_values(cls.Fields.DATE)
        return cls(data)

    @classmethod
    def verify(cls, data):
        # all county level us data must have a fips code
        county_level = data[cls.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        is_us = data[cls.Fields.COUNTRY] == "USA"
        missing_fips = data[cls.Fields.FIPS].isnull()
        us_missing_fips = data[county_level & is_us & missing_fips]

    @classmethod
    def _fill_missing_state_with_county(cls, data):
        state_groupby_fields = [
            cls.Fields.DATE,
            cls.Fields.COUNTRY,
            cls.Fields.SOURCE,
            cls.Fields.STATE,
        ]

        county_data = data[data.aggregate_level == "county"]
        state_data = county_data.groupby(state_groupby_fields).sum().reset_index()
        state_data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.STATE.value
        state_data = state_data.set_index(state_groupby_fields)
        state_data[cls.Fields.GENERATED] = True

        existing_state = data[data.aggregate_level == AggregationLevel.STATE.value]
        existing_state = existing_state.set_index(state_groupby_fields)

        non_matching = state_data[~state_data.index.isin(existing_state.index)]
        return pd.concat([data, non_matching.reset_index()])
