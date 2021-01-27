from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class UsaFactsDataSource(data_source.DataSource):
    SOURCE_NAME = "USAFacts"

    COMMON_DF_CSV_PATH = "data/cases-covid-county-data/timeseries-usafacts.csv"

    EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]
