import logging
import numpy
import pandas as pd
import pathlib

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.dataset_utils import AggregationLevel
from libs import enums
from libs.us_state_abbrev import ABBREV_US_UNKNOWN_COUNTY_FIPS

_logger = logging.getLogger(__name__)


class JHUDataset(data_source.DataSource):
    """JHU Timeseries Data.

    Originally available at:
    https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports
    """

    DATA_FOLDER = "data/cases-jhu/csse_covid_19_daily_reports"
    SOURCE_NAME = "JHU"
    HAS_AGGREGATED_NYC_BOROUGH = True

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
    INDEX_FIELD_MAP = {
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }
    COMMON_FIELD_MAP = {
        CommonFields.CASES: Fields.CONFIRMED,
        CommonFields.DEATHS: Fields.DEATHS,
        CommonFields.RECOVERED: Fields.RECOVERED,
    }

    def __init__(self, input_dir):
        loaded_data = []
        for path in sorted(input_dir.glob("*.csv")):
            date = path.stem
            data = pd.read_csv(path, dtype={"FIPS": str})
            data = data.rename(columns=self.RENAMED_COLUMNS)
            data[self.Fields.DATE] = pd.to_datetime(date)
            loaded_data.append(data)

        data = pd.concat(loaded_data)
        data = self.standardize_data(data)
        super().__init__(data)

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
        data = data.replace({cls.Fields.COUNTRY: country_remap})
        states = data[cls.Fields.STATE].apply(dataset_utils.parse_state)

        county_from_state = data[cls.Fields.STATE].apply(dataset_utils.parse_county_from_state)
        data[cls.Fields.COUNTY] = data[cls.Fields.COUNTY].combine_first(county_from_state)
        data[cls.Fields.STATE] = states
        data = cls._fill_incomplete_county_data(data)

        state_only = data[cls.Fields.FIPS].isnull() & data[cls.Fields.COUNTY].isnull()
        # Pad fips values to 5 spots
        data[cls.Fields.FIPS] = data[cls.Fields.FIPS].apply(
            lambda x: f"{x.zfill(5)}" if type(x) == str else x
        )

        data[cls.Fields.AGGREGATE_LEVEL] = numpy.where(state_only, "state", "county")
        data = cls._aggregate_fips_data(data)

        dataset_utils.assert_counties_have_fips(
            data, county_key=cls.Fields.COUNTY, fips_key=cls.Fields.FIPS
        )

        return data

    @classmethod
    def _fill_incomplete_county_data(cls, data):
        """Fills incomplete county data.

        Most of this data is "unassigned" (at least in more recent days.
        """
        overrides = {
            # Assigning nantucket county to dukes and nantucket
            ("MA", "Dukes and Nantucket"): "25019"
        }
        for (state, county), fips in overrides.items():
            matches_state = data[cls.Fields.STATE] == state
            matches_county = data[cls.Fields.COUNTY] == county
            data.loc[matches_state & matches_county, cls.Fields.FIPS] = fips

        has_county = data[cls.Fields.COUNTY].notnull()
        rows_to_replace = has_county & data[cls.Fields.FIPS].isnull()
        data.loc[rows_to_replace, cls.Fields.FIPS] = data.loc[
            rows_to_replace, cls.Fields.STATE
        ].map(ABBREV_US_UNKNOWN_COUNTY_FIPS)
        still_no_fips = has_county & data[cls.Fields.FIPS].isnull()
        if still_no_fips.any():
            _logger.warning(
                f"County rows without FIPS: {data.loc[still_no_fips, [cls.Fields.STATE, cls.Fields.COUNTY]]}"
            )
        return data

    @classmethod
    def _aggregate_fips_data(cls, data):
        is_county_level = data[cls.Fields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
        has_fips = data[cls.Fields.FIPS].notnull()
        county_data = data[is_county_level & has_fips]

        # This key aggregates county level data based on the fips code.
        # note that we are not using county name to aggregate. There are some rows in the
        # JHU data that have multiple different county names per fips code.
        # Right now, we are just combining that data, it may be more appropriate to take the max.
        group_key = [
            cls.Fields.DATE,
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.FIPS,
            cls.Fields.STATE,
            cls.Fields.COUNTRY,
        ]
        result = county_data.groupby(group_key).sum().reset_index()

        return pd.concat([data[~(is_county_level & has_fips)], result])

    @classmethod
    def local(cls) -> "JHUDataset":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_FOLDER)

    @classmethod
    def latest_path(cls) -> pathlib.Path:
        """Finds the path to the most recent JHU csv."""
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        jhu_dir = data_root / cls.DATA_FOLDER
        latest_date, latest_path = max((path.stem, path) for path in jhu_dir.glob("*.csv"))
        return latest_path
