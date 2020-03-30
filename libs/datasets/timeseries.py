from typing import List, Iterator
import enum
import pandas as pd
import datetime
from libs.datasets import data_source
from libs.datasets import AggregationLevel


class TimeseriesDataset(object):
    class Fields(object):
        DATE = "date"
        COUNTRY = "country"
        STATE = "state"
        COUNTY = "county"
        CASES = "cases"
        DEATHS = "deaths"
        RECOVERED = "recovered"
        # Name of source of dataset, i.e. JHU
        SOURCE = "source"
        IS_SYNTHETIC = "is_synthetic"
        AGGREGATE_LEVEL = "aggregate_level"
        GENERATED = 'generated'

    def __init__(self, data: pd.DataFrame):
        self.data = data

    @property
    def states(self) -> List:
        return self.data[self.Fields.STATE].dropna().unique().tolist()

    @property
    def data_by_state(self) -> Iterator:
        state_timeseries = self.get_aggregation_level(AggregationLevel.STATE)
        for state in state_timeseries.states:
            yield state, state_timeseries.get_dataframe(state=state)

    def get_subset(
        self, aggregation_level, on=None, after=None, country=None, state=None, county=None
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

        if on:
            data = data[data.date == on]
        if after:
            data = data[data.date > after]

        return self.__class__(data)

    def get_data(self, country=None, state=None, county=None) -> pd.DataFrame:
        data = self.data
        if country:
            data = data[data.country == country]
        if state:
            data = data[data.state == state]
        if county:
            data = data[data.county == county]
        return data

    @classmethod
    def from_source(cls, source: data_source.DataSource, fill_missing_state=True):
        """Loads data from a specific datasource."""
        if not source.TIMESERIES_FIELD_MAP:
            raise ValueError("Source must have field timeseries field map.")

        data = source.data
        to_common_fields = {value: key for key, value in source.TIMESERIES_FIELD_MAP.items()}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        data[cls.Fields.SOURCE] = source.SOURCE_NAME
        data[cls.Fields.GENERATED] = False
        if fill_missing_state:
            data = cls._fill_missing_state_with_county(data)

        # Choosing to sort by date
        data = data.sort_values(cls.Fields.DATE)
        return cls(data)

    @classmethod
    def _fill_missing_state_with_county(cls, data):
        state_groupby_fields = [cls.Fields.DATE, cls.Fields.COUNTRY, cls.Fields.SOURCE, cls.Fields.STATE]

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
