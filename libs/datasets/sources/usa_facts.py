from functools import lru_cache

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df

from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries


class UsaFactsDataSource(data_source.DataSource):
    COMMON_DF_CSV_PATH = "data/cases-covid-county-data/timeseries-usafacts.csv"

    SOURCE_NAME = "USAFacts"

    EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]
