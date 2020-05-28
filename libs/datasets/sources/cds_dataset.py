import logging
import numpy
import pandas as pd
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields

_logger = logging.getLogger(__name__)


def fill_missing_county_with_city(row):
    """Fills in missing county data with city if available.

    """
    if pd.isnull(row.county) and not pd.isnull(row.city):
        if row.city == "New York City":
            return "New York"
        return row.city

    return row.county


class CDSDataset(data_source.DataSource):
    DATA_PATH = "data/cases-cds/timeseries.csv"
    SOURCE_NAME = "CDS"

    class Fields(object):
        CITY = "city"
        COUNTY = "county"
        STATE = "state"
        COUNTRY = "country"
        POPULATION = "population"
        LATITUDE = "lat"
        LONGITUDE = "long"
        URL = "url"
        CASES = "cases"
        DEATHS = "deaths"
        RECOVERED = "recovered"
        ACTIVE = "active"
        TESTED = "tested"
        GROWTH_FACTOR = "growthFactor"
        DATE = "date"
        AGGREGATE_LEVEL = "aggregate_level"
        FIPS = "fips"
        NEGATIVE_TESTS = "negative_tests"
        HOSPITALIZED = "hospitalized"
        ICU = "icu"

    INDEX_FIELD_MAP = {
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.CASES: Fields.CASES,
        CommonFields.POSITIVE_TESTS: Fields.CASES,
        CommonFields.NEGATIVE_TESTS: Fields.NEGATIVE_TESTS,
        CommonFields.POPULATION: Fields.POPULATION,
        CommonFields.CUMULATIVE_ICU: Fields.ICU,
        CommonFields.CUMULATIVE_HOSPITALIZED: Fields.HOSPITALIZED,
    }

    TEST_FIELDS = [
        Fields.COUNTRY,
        Fields.STATE,
        Fields.FIPS,
        Fields.DATE,
        Fields.CASES,
        Fields.TESTED,
    ]

    COMMON_TEST_FIELDS = [
        CommonFields.COUNTRY,
        CommonFields.STATE,
        CommonFields.FIPS,
        CommonFields.DATE,
        CommonFields.POSITIVE_TESTS,
        CommonFields.NEGATIVE_TESTS,
    ]

    def __init__(self, input_path):
        data = pd.read_csv(
            input_path,
            parse_dates=[self.Fields.DATE],
            dtype={self.Fields.FIPS: str},
            low_memory=False,
        )
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def local(cls) -> "CDSDataset":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_PATH)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data = dataset_utils.strip_whitespace(data)

        data = cls.remove_duplicate_city_data(data)

        # CDS state level aggregates are identifiable by not having a city or county.
        only_county = data[cls.Fields.COUNTY].notnull() & data[cls.Fields.STATE].notnull()
        county_hits = numpy.where(only_county, "county", None)
        only_state = (
            data[cls.Fields.COUNTY].isnull()
            & data[cls.Fields.CITY].isnull()
            & data[cls.Fields.STATE].notnull()
        )
        only_country = (
            data[cls.Fields.COUNTY].isnull()
            & data[cls.Fields.CITY].isnull()
            & data[cls.Fields.STATE].isnull()
            & data[cls.Fields.COUNTRY].notnull()
        )

        state_hits = numpy.where(only_state, "state", None)
        county_hits[state_hits != None] = state_hits[state_hits != None]
        county_hits[only_country] = "country"
        data[cls.Fields.AGGREGATE_LEVEL] = county_hits

        # Backfilling FIPS data based on county names.
        # The following abbrev mapping only makes sense for the US
        # TODO: Fix all missing cases
        data = data[data["country"] == "United States"]
        data[CommonFields.COUNTRY] = "USA"
        data[CommonFields.STATE] = data[cls.Fields.STATE].apply(
            lambda x: US_STATE_ABBREV[x] if x in US_STATE_ABBREV else x
        )

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_fips_using_county(data, fips_data)
        no_fips = data[CommonFields.FIPS].isna()
        if no_fips.sum() > 0:
            logging.error(f"Removing rows without fips id: {str(data.loc[no_fips])}")
            data = data.loc[~no_fips]

        data.set_index(["date", "fips"], inplace=True)
        if data.index.has_duplicates:
            # Use keep=False when logging so the output contains all duplicated rows, not just the first or last
            # instance of each duplicate.
            logging.error(f"Removing duplicates: {str(data.index.duplicated(keep=False))}")
            data = data.loc[~data.index.duplicated(keep=False)]
        data.reset_index(inplace=True)

        # ADD Negative tests
        data[cls.Fields.NEGATIVE_TESTS] = data[cls.Fields.TESTED] - data[cls.Fields.CASES]

        return data

    @classmethod
    def remove_duplicate_city_data(cls, data):
        # City data before 3-23 was not duplicated, copy the city name to the county field.
        select_pre_march_23 = data.date < "2020-03-23"
        data.loc[select_pre_march_23, cls.Fields.COUNTY] = data.loc[select_pre_march_23].apply(
            fill_missing_county_with_city, axis=1
        )
        # Don't want to return city data because it's duplicated in county
        return data.loc[
            select_pre_march_23 | ((~select_pre_march_23) & data[cls.Fields.CITY].isnull())
        ]
