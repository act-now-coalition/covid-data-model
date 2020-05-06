from typing import List
import logging
import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields


class NYTimesDataset(data_source.DataSource):
    DATA_URL = "https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv"
    SOURCE_NAME = "NYTimes"

    HAS_AGGREGATED_NYC_BOROUGH = True

    class Fields(object):
        DATE = "date"
        COUNTY = "county"
        STATE = "state"
        FIPS = "fips"
        COUNTRY = "country"
        AGGREGATE_LEVEL = "aggregate_level"
        CASES = "cases"
        DEATHS = "deaths"

    INDEX_FIELD_MAP = {
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }
    COMMON_FIELD_MAP = {
        TimeseriesDataset.Fields.CASES: Fields.CASES,
        TimeseriesDataset.Fields.DEATHS: Fields.DEATHS,
    }

    def __init__(self, input_path):
        data = pd.read_csv(input_path, parse_dates=[self.Fields.DATE], dtype={"fips": str})
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def load(cls) -> "NYTimesDataset":
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
