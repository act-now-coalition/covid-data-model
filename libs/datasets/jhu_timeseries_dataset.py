import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import dataset_utils


class JHUTimeseriesData(object):
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

    RENAMED_COLUMNS = {
        "Country/Region": Fields.COUNTRY_REGION,
        "Province/State": Fields.STATE,
        "Lat": Fields.LATITUDE,
        "Long_": Fields.LONGITUDE,
        "Last Update": Fields.LAST_UPDATE,
    }

    COMMON_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY_REGION,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.COUNTY: Fields.COUNTY,
        TimeseriesDataset.Fields.CASES: Fields.CONFIRMED,
        TimeseriesDataset.Fields.DEATHS: Fields.DEATHS,
        TimeseriesDataset.Fields.RECOVERED: Fields.RECOVERED,
    }

    def __init__(self, input_dir):
        loaded_data = []
        for path in input_dir.glob("*.csv"):
            date = path.stem
            data = pd.read_csv(path)
            data = data.rename(columns=self.RENAMED_COLUMNS)
            data[self.Fields.DATE] = pd.Timestamp(date)
            loaded_data.append(data)

        data = pd.concat(loaded_data)
        self.data = self.standardize_data(data)

    def verify(self):

        pass

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

        return data

    def to_common(self, state_only=False, county_only=False) -> TimeseriesDataset:

        data = self.data
        if state_only:
            data = data[data[self.Fields.FIPS].isnull() & data[self.Fields.COUNTY].isnull()]

        if county_only:
            data = data[data[self.Fields.COUNTY].notnull() | data[self.Fields.FIPS].notnull()]

        to_common_fields = {value: key for key, value in self.COMMON_FIELD_MAP.items()}
        final_columns = to_common_fields.values()

        data = data.rename(columns=to_common_fields)[final_columns]
        data[TimeseriesDataset.Fields.SOURCE] = self.SOURCE_NAME

        return TimeseriesDataset(data)

    @classmethod
    def build_from_local_github(cls) -> "JHUTimeseriesData":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_FOLDER)
