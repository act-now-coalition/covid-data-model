from typing import Dict, List, Tuple
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs import enums
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.common_fields import CommonIndexFields
from libs.us_state_abbrev import ABBREV_US_UNKNOWN_COUNTY_FIPS

BACKFILLED_CASES = [
    # On 2020-07-24, CT reported a backfill of 440 additional positive cases.
    # https://portal.ct.gov/Office-of-the-Governor/News/Press-Releases/2020/07-2020/Governor-Lamont-Coronavirus-Update-July-24
    ("09", "2020-07-24", 440),
    # https://portal.ct.gov/Office-of-the-Governor/News/Press-Releases/2020/07-2020/Governor-Lamont-Coronavirus-Update-July-29
    ("09", "2020-07-29", 384),
]


def _calculate_county_adjustments(
    data: pd.DataFrame, date: str, backfilled_cases: int, state_fips: str
) -> Dict[str, int]:
    """Calculating number of cases to remove per county, weighted on number of new cases per county.

    Weighting on number of new cases per county gives a reasonable measure of where the backfilled
    cases ended up.

    Args:
        data: Input Data.
        date: Date of backfill.
        backfilled_cases: Number of backfilled cases.
        state_fips: FIPS code for state.

    Returns: Dictionary of estimated fips -> backfilled cases.
    """
    is_state = data[CommonFields.FIPS].str.match(f"{state_fips}[0-9][0-9][0-9]")
    is_not_unknown = data[CommonFields.FIPS] != f"{state_fips}999"

    fields = [CommonFields.DATE, CommonFields.FIPS, CommonFields.CASES]
    cases = (
        data.loc[is_state & is_not_unknown, fields]
        .set_index([CommonFields.FIPS, CommonFields.DATE])
        .sort_index()
    )
    cases = cases.diff().reset_index(level=1)
    cases_on_date = cases[cases.date == date]["cases"]
    # For states with more counties, rounding could lead to the sum of the counties diverging from
    # the backfilled cases count.
    return (cases_on_date / cases_on_date.sum() * backfilled_cases).round().to_dict()


def remove_backfilled_cases(
    data: pd.DataFrame, backfilled_cases: List[Tuple[str, str, int]]
) -> pd.DataFrame:
    """Removes reported backfilled cases from case totals.

    Args:
        data: Data
        backfilled_cases: List of backfilled case info.

    Returns: Updated data frame.
    """
    for state_fips, date, cases in backfilled_cases:
        adjustments = _calculate_county_adjustments(data, date, cases, state_fips)
        is_on_or_after_date = data[CommonFields.DATE] >= date
        for fips, count in adjustments.items():
            is_fips_data_after_date = is_on_or_after_date & (data[CommonFields.FIPS] == fips)
            data.loc[is_fips_data_after_date, CommonFields.CASES] -= int(count)

    return data


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

        data = remove_backfilled_cases(data, BACKFILLED_CASES)

        return data
