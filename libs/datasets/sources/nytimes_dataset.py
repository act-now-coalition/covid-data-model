import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs import enums
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.common_fields import CommonIndexFields
from libs.us_state_abbrev import ABBREV_US_UNKNOWN_COUNTY_FIPS

# On 2020-07-24, CT reported a backfill of 440 additional positive cases.
# https://portal.ct.gov/Office-of-the-Governor/News/Press-Releases/2020/07-2020/Governor-Lamont-Coronavirus-Update-July-24
# See https://trello.com/c/rPiM6UF4/333-adjust-ct-case-data-based-on-reported-backfill-vs-current-cases
# for more information on backfill.
ESTIMATED_REMOVED_CT_CASES_COUNTY = {
    "09001": 188,
    "09003": 154,
    "09005": 24,
    "09007": 4,
    "09009": 39,
    "09011": 8,
    "09013": 22,
    "09015": 2,
}


class NYTimesDataset(data_source.DataSource):
    SOURCE_NAME = "NYTimes"

    DATA_FOLDER = "data/cases-nytimes"
    COUNTIES_DATA_FILE = "us-counties.csv"

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
        CommonFields.CASES: Fields.CASES,
        CommonFields.DEATHS: Fields.DEATHS,
    }

    def __init__(self, input_path):
        data = pd.read_csv(input_path, parse_dates=[self.Fields.DATE], dtype={"fips": str})
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def local(cls) -> "NYTimesDataset":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_FOLDER / cls.COUNTIES_DATA_FILE)

    @classmethod
    def _remove_ct_cases(cls, data: pd.DataFrame) -> pd.DataFrame:
        is_county = data[cls.Fields.AGGREGATE_LEVEL] == "county"
        is_ct = data[cls.Fields.STATE] == "CT"
        # July 24th was the day there was a backfill of cases
        is_on_or_after_july_24 = data[cls.Fields.DATE] >= "2020-07-24"
        is_ct_county_data_after_jul_24 = is_county & is_ct & is_on_or_after_july_24

        # Remove backfilled tests from cumulatives on and after July 24th.
        for fips, count in ESTIMATED_REMOVED_CT_CASES_COUNTY.items():
            is_ct_fips_data_after_jul_24 = is_ct_county_data_after_jul_24 & (
                data[CommonFields.FIPS] == fips
            )
            data.loc[is_ct_fips_data_after_jul_24, CommonFields.CASES] -= count
        return data

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data[cls.Fields.COUNTRY] = "USA"
        data[cls.Fields.AGGREGATE_LEVEL] = "county"
        data = dataset_utils.strip_whitespace(data)
        data[cls.Fields.STATE] = data[cls.Fields.STATE].apply(dataset_utils.parse_state)
        # Super hacky way of filling in new york.
        data.loc[data[cls.Fields.COUNTY] == "New York City", "county"] = "New York County"
        data.loc[data[cls.Fields.COUNTY] == "New York County", "fips"] = "36061"

        # UNKNOWN_FIPS is overwritten with values from ABBREV_US_UNKNOWN_COUNTY_FIPS below.
        data.loc[data[cls.Fields.COUNTY] == "Unknown", "fips"] = enums.UNKNOWN_FIPS

        # https://github.com/nytimes/covid-19-data/blob/master/README.md#geographic-exceptions
        # Both Joplin and Kansas City numbers are reported separately from the surrounding counties.
        # Until we figure out a better way to spread amongst counties they are in, combining all
        # data missing a fips into one unknown fips.
        is_kc = data[cls.Fields.COUNTY] == "Kansas City"
        is_joplin = data[cls.Fields.COUNTY] == "Joplin"
        data.loc[is_kc | is_joplin, cls.Fields.FIPS] = enums.UNKNOWN_FIPS
        is_missouri = data[cls.Fields.STATE] == "MO"
        is_unknown = data[cls.Fields.FIPS] == enums.UNKNOWN_FIPS
        missouri_unknown = data.loc[is_missouri & is_unknown, :]
        group_columns = [
            cls.Fields.AGGREGATE_LEVEL,
            cls.Fields.FIPS,
            cls.Fields.DATE,
            cls.Fields.COUNTRY,
            cls.Fields.STATE,
        ]
        missouri_unknown = missouri_unknown.groupby(group_columns).sum().reset_index()
        missouri_unknown[cls.Fields.COUNTY] = "Aggregated City and Unknown Data"
        data = pd.concat([data.loc[~(is_missouri & is_unknown), :], missouri_unknown])

        # Change all the 99999 FIPS to per-state unknown
        unknown_fips = data[cls.Fields.FIPS] == enums.UNKNOWN_FIPS
        data.loc[unknown_fips, cls.Fields.FIPS] = data.loc[unknown_fips, cls.Fields.STATE].map(
            ABBREV_US_UNKNOWN_COUNTY_FIPS
        )

        data = cls._remove_ct_cases(data)

        return data
