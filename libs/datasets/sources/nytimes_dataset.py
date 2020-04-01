from typing import List
import logging
import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils


class NYTimesDataset(data_source.DataSource):
    DATA_URL = "https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv"
    SOURCE_NAME = "NYTimes"

    class Fields(object):
        DATE = "date"
        COUNTY = "county"
        STATE = "state"
        FIPS = "fips"
        CASES = "cases"
        DEATHS = "deaths"

        COUNTRY = "country"
        AGGREGATE_LEVEL = "aggregate_level"

    TIMESERIES_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.FIPS: Fields.FIPS,
        TimeseriesDataset.Fields.CASES: Fields.CASES,
        TimeseriesDataset.Fields.DEATHS: Fields.DEATHS,
        TimeseriesDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    def __init__(self, input_path):
        data = pd.read_csv(input_path, parse_dates=[self.Fields.DATE], dtype={"fips": str})
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def load(cls) -> "CDSTimeseriesData":
        return cls(cls.DATA_URL)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data[cls.Fields.COUNTRY] = "USA"
        data = dataset_utils.strip_whitespace(data)
        data[cls.Fields.STATE] = data[cls.Fields.STATE].apply(dataset_utils.parse_state)
        # Super hacky way of filling in new york.
        data.loc[data[cls.Fields.COUNTY] == 'New York City', 'county'] = 'New York County'
        data.loc[data[cls.Fields.COUNTY] == 'New York County', 'fips'] = '36061'
        data[cls.Fields.AGGREGATE_LEVEL] = "county"
        return data
