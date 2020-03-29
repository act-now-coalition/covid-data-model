from typing import List
import logging
import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils


class NYTimesTimeseriesData(data_source.DataSource):
    DATA_URL = 'https://github.com/nytimes/covid-19-data/raw/6cb66d9a821ce8225f6f9ffcb77ce6db9889c14c/us-counties.csv'
    SOURCE_NAME = "NYTimes"

    class Fields(object):
        DATE = "date"
        COUNTY = "county"
        STATE = "state"
        FIPS = 'fips'
        CASES = "cases"
        DEATHS = "deaths"

        COUNTRY = 'country'
        AGGREGATE_LEVEL = 'aggregate_level'


    COMMON_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.COUNTY: Fields.COUNTY,
        TimeseriesDataset.Fields.CASES: Fields.CASES,
        TimeseriesDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL
    }

    def __init__(self, input_path):
        data = pd.read_csv(input_path, parse_dates=[self.Fields.DATE])
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def build_from_url(cls) -> "CDSTimeseriesData":
        return cls(cls.DATA_URL)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data[cls.Fields.COUNTRY] = 'USA'
        data = dataset_utils.strip_whitespace(data)
        states = data[cls.Fields.STATE].apply(dataset_utils.parse_state)
        data[cls.Fields.AGGREGATE_LEVEL] = 'county'

        return data
