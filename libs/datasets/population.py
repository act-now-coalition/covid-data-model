from typing import List
import logging
import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import AggregationLevel


class PopulationDataset(object):
    class Fields(object):
        COUNTRY = "country"
        STATE = "state"
        POPULATION = "population"
        FIPS = "fips"
        # Name of source of dataset, i.e. JHU
        SOURCE = "source"
        AGGREGATE_LEVEL = "aggregate_level"
        GENERATED = "generated"
        COUNTY = "county"

    def __init__(self, data):
        self.data = data

    @classmethod
    def from_source(cls, source: data_source.DataSource, fill_missing_state=True):
        """Loads data from a specific datasource."""
        if not source.POPULATION_FIELD_MAP:
            raise ValueError("Source must have beds field map.")

        data = source.data
        to_common_fields = {
            value: key for key, value in source.POPULATION_FIELD_MAP.items()
        }
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)
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

        data[cls.Fields.POPULATION] = data[cls.Fields.POPULATION].fillna(0)
        return cls(data)

    def get_state_level(self, country, state):
        data = dataset_utils.get_state_level_data(self.data, country, state).population
        if len(data):
            return data.iloc[0]
        return None

    def get_county_level(self, country, state, county):
        data = dataset_utils.get_county_level_data(
            self.data, country, state, county
        ).population
        if len(data):
            return data.iloc[0]
        return None
