from typing import List
import logging
import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import custom_aggregations


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
    def from_source(cls, source: "DataSource", fill_missing_state=True):
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
        group = [cls.Fields.SOURCE, cls.Fields.AGGREGATE_LEVEL, cls.Fields.STATE, cls.Fields.COUNTRY, cls.Fields.GENERATED]
        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=False
        )

        if fill_missing_state:
            state_groupby_fields = [
                cls.Fields.SOURCE,
                cls.Fields.COUNTRY,
                cls.Fields.STATE,
            ]
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

    def get_county_level(self, country, state, county=None, fips=None):
        if not (county or fips) or (county and fips):
            raise ValueError("Must only pass fips or county")
        data = dataset_utils.get_county_level_data(
            self.data, country, state, county=county, fips=fips
        ).population
        if len(data):
            return data.iloc[0]
        return None
