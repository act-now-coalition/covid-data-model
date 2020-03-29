from typing import List
import logging
import numpy
import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils

_logger = logging.getLogger(__name__)


def fill_missing_county_with_city(row):
    """Fills in missing county data with city if available.

    """
    if pd.isnull(row.county) and not pd.isnull(row.city):
        if row.city == "New York City":
            return "New York"
        return row.city

    return row.county


def check_uniqueness(data: pd.DataFrame, group: List[str], field: str):
    """Logs warning if more than one instance of data when grouped on `group`.

    Args:
        data: DataFrame
        group: List of columns to group on
        field: Column to check for
    """
    date_country_count = data.groupby(group).count()
    non_unique_countries = date_country_count[field] > 1
    if sum(non_unique_countries) > 1:
        _logger.warning(
            f"Found {sum(non_unique_countries)} records that "
            "have non unique date-country records."
        )


class CDSTimeseriesData(data_source.DataSource):
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

    COMMON_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.COUNTY: Fields.COUNTY,
        TimeseriesDataset.Fields.CASES: Fields.CASES,
        TimeseriesDataset.Fields.DEATHS: Fields.DEATHS,
        TimeseriesDataset.Fields.RECOVERED: Fields.RECOVERED,
    }

    def __init__(self, input_path):
        data = pd.read_csv(input_path, parse_dates=[self.Fields.DATE])
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def build_from_local_github(cls) -> "CDSTimeseriesData":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_PATH)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data = dataset_utils.strip_whitespace(data)

        data[cls.Fields.COUNTY] = dataset_utils.standardize_county(
            data[cls.Fields.COUNTY]
        )

        # Don't want to return city data because it's duplicated in county
        # City data before 3-23 was not duplicated.
        # data = data[data[cls.Fields.CITY].isnull()]
        pre_march_23 = data[data.date < "2020-03-23"]
        pre_march_23.county = pre_march_23.apply(fill_missing_county_with_city, axis=1)
        split_data = [
            pre_march_23,
            data[(data.date >= "2020-03-23") & data[cls.Fields.CITY].isnull()],
        ]
        data = pd.concat(split_data)

        # CDS state level aggregates are identifiable by not having a city or county.
        only_state = (
            data[cls.Fields.COUNTY].isnull()
            & data[cls.Fields.CITY].isnull()
            & data[cls.Fields.STATE].notnull()
        )
        data[cls.Fields.AGGREGATE_LEVEL] = numpy.where(only_state, "state", "county")
        return data
