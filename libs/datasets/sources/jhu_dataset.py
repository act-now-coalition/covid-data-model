import logging
import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets.common_fields import CommonFields
from libs.datasets.dataset_utils import AggregationLevel
from libs import enums
_logger = logging.getLogger(__name__)


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
        COUNTRY = "Country_Region"
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
        "Country/Region": Fields.COUNTRY,
        "Province/State": Fields.STATE,
        "Lat": Fields.LATITUDE,
        "Long_": Fields.LONGITUDE,
        "Last Update": Fields.LAST_UPDATE,
    }

    COMMON_FIELD_MAP = {
        Fields.AGGREGATE_LEVEL: CommonFields.AGGREGATE_LEVEL,
        Fields.COUNTRY: CommonFields.COUNTRY,
        Fields.COUNTY: CommonFields.COUNTY,
        Fields.STATE: CommonFields.STATE,
        Fields.FIPS: CommonFields.FIPS,
    }

    @classmethod
    def apply_source_specific_cleaning(cls, data: pd.DataFrame) -> pd.DataFrame:
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
        data = data.replace({CommonFields.COUNTRY: country_remap})
        states = data[CommonFields.STATE].apply(dataset_utils.parse_state)
        county_from_state = data[CommonFields.STATE].apply(
            dataset_utils.parse_county_from_state
        )
        data[CommonFields.COUNTY] = data[CommonFields.COUNTY].combine_first(
            county_from_state
        )
        data[CommonFields.STATE] = states
        data = cls._fill_incomplete_county_data(data)

        state_only = data[cls.Fields.FIPS].isnull() & data[cls.Fields.COUNTY].isnull()
        # Pad fips values to 5 spots
        data[CommonFields.FIPS] = data[CommonFields.FIPS].apply(
            lambda x: f"{x.zfill(5)}" if type(x) == str else x
        )
        data[cls.Fields.AGGREGATE_LEVEL] = numpy.where(state_only, "state", "county")
        return data

    @classmethod
    def _fill_incomplete_county_data(cls, data):
        """Fills incomplete county data.

        Most of this data is "unassigned" (at least in more recent days.
        We probably need to either give each state its own fake FIP, or spread this
        out over the existing counties for the state.
        """
        overrides = {
            # Assigning nantucket county to dukes and nantucket
            ('MA', 'Dukes and Nantucket'): '25019'
        }
        for (state, county), fips in overrides.items():
            matches_state = data[cls.Fields.STATE] == state
            matches_county = data[cls.Fields.COUNTY] == county
            data.loc[matches_state & matches_county, cls.Fields.FIPS] = fips

        has_county = data[cls.Fields.COUNTY].notnull()
        rows_to_replace = has_county & data[cls.Fields.FIPS].isnull()
        data.loc[rows_to_replace, cls.Fields.FIPS] = enums.UNKNOWN_FIPS
        return data

    @classmethod
    def filter_and_aggregate_duplicates(cls, data):
        is_county_level = data[cls.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        has_fips = data[cls.Fields.FIPS].notnull()
        county_data = data[is_county_level & has_fips]

        # This key aggregates county level data based on the fips code.
        # note that we are not using county name to aggregate. There are some rows in the
        # JHU data that have multiple different county names per fips code.
        # Right now, we are just combining that data, it may be more appropriate to take the max.
        group_key = [
            cls.Fields.DATE,
            CommonFields.AGGREGATE_LEVEL,
            CommonFields.FIPS,
            CommonFields.STATE,
            CommonFields.COUNTRY,
        ]
        result = county_data.groupby(group_key).sum().reset_index()

        return pd.concat([
            data[~(is_county_level & has_fips)],
            result
        ])

    @classmethod
    def local(cls) -> "JHUTimeseriesData":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_dir = data_root / cls.DATA_FOLDER
        loaded_data = []
        for path in sorted(input_dir.glob("*.csv")):
            date = path.stem
            data = pd.read_csv(path, dtype={"FIPS": str})
            data = data.rename(columns=self.RENAMED_COLUMNS)
            data[self.Fields.DATE] = pd.to_datetime(date)
            loaded_data.append(data)

        data = pd.concat(loaded_data)
        return cls(data)
