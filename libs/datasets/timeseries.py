import pandas as pd
import datetime


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

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_country(self, country) -> "TimeseriesDataset":
        return self.__class__(self.data[self.data.country == country])

    def get_date(self, on=None, after=None) -> "TimeseriesDataset":
        if on:
            return self.__class__(self.data[self.data.date == on])
        if after:
            return self.__class__(self.data[self.data.date > after])

        return self
