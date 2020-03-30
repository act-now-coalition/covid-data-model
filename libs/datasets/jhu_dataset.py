import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import dataset_utils
from libs.datasets import data_source


class JHUDataset(data_source.DataSource):
    """JHU Timeseries Data.

    Originally available at:
    https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports

    """

    DATA_FOLDER = "data/cases-jhu/csse_covid_19_daily_reports"
    SOURCE_NAME = "JHU"

    class Fields(object):
        FIPS = "FIPS"
        COUNTY = "Admin2"
        STATE = "Province_State"
        COUNTRY_REGION = "Country_Region"
        LAST_UPDATE = "Last_Update"
        LATITUDE = "Latitude"
        LONGITUDE = "Longitude"
        CONFIRMED = "Confirmed"
        DEATHS = "Deaths"
        RECOVERED = "Recovered"
        ACTIVE = "Active"
        COMBINED_KEY = "Combined_Key"
        # Added manually in init.
        DATE = "date"
        AGGREGATE_LEVEL = "aggregate_level"

    RENAMED_COLUMNS = {
        "Country/Region": Fields.COUNTRY_REGION,
        "Province/State": Fields.STATE,
        "Lat": Fields.LATITUDE,
        "Long_": Fields.LONGITUDE,
        "Last Update": Fields.LAST_UPDATE,
    }

    TIMESERIES_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY_REGION,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.COUNTY: Fields.COUNTY,
        TimeseriesDataset.Fields.CASES: Fields.CONFIRMED,
        TimeseriesDataset.Fields.DEATHS: Fields.DEATHS,
        TimeseriesDataset.Fields.RECOVERED: Fields.RECOVERED,
        TimeseriesDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL
    }

    def __init__(self, input_dir):
        loaded_data = []
        for path in sorted(input_dir.glob("*.csv")):
            date = path.stem
            data = pd.read_csv(path)
            data = data.rename(columns=self.RENAMED_COLUMNS)
            data[self.Fields.DATE] = pd.to_datetime(date)
            loaded_data.append(data)
            # if date == '03-26-2020':
            #     print(data)

        data = pd.concat(loaded_data)
        self.data = self.standardize_data(data)
        # print(self.data.columns)
        recent = self.data[self.data.date == '03-26-2020']


    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data = dataset_utils.strip_whitespace(data)
        # TODO Figure out how to rename to some ISO standard.
        country_remap = {
            "Mainland China": "China",
            "Bahamas, The": "Bahamas",
            "Congo (Brazzaville)": "Congo",
            "Congo (Kinshasa)": "Congo",
            "Diamond Princess": "Cruise Ship",
            "Hong Kong SAR": "Hong Kong",
            "Iran (Islamic Republic of)": "Iran",
            "Korea, South": "South Korea",
            "Taiwan*": "Taiwan",
            "UK": "United Kingdom",
            "US": "USA",
        }
        data = data.replace({cls.Fields.COUNTRY_REGION: country_remap})

        states = data[cls.Fields.STATE].apply(dataset_utils.parse_state)
        county_from_state = data[cls.Fields.STATE].apply(
            dataset_utils.parse_county_from_state
        )
        data[cls.Fields.COUNTY] = data[cls.Fields.COUNTY].combine_first(
            county_from_state
        )
        data[cls.Fields.STATE] = states
        state_only = data[cls.Fields.FIPS].isnull() & data[cls.Fields.COUNTY].isnull()
        data[cls.Fields.AGGREGATE_LEVEL] = numpy.where(state_only, "state", "county")
        return data

    @classmethod
    def build_from_local_github(cls) -> "JHUTimeseriesData":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_FOLDER)
