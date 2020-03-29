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

        @classmethod
        def build_row(cls, date: datetime.datetime, country: str, **updates):
            """Builds a new row, applying updates."""
            row = {
                cls.DATE: date,
                cls.COUNTRY: country,
                cls.STATE: None,
                cls.COUNTY: None,
                cls.DEATHS: None,
                cls.RECOVERED: None,
                cls.IS_SYNTHETIC: False,
            }
            row.update(updates)
            return row

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
