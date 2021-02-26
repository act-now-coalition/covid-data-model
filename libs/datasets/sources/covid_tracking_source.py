from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets.taglib import UrlStr


class CovidTrackingDataSource(data_source.DataSource):
    SOURCE_TYPE = "covid_tracking"
    SOURCE_NAME = "The COVID Tracking Project"
    SOURCE_URL = UrlStr("https://covidtracking.com/")

    COMMON_DF_CSV_PATH = "data/covid-tracking/timeseries.csv"

    EXPECTED_FIELDS = [
        CommonFields.DEATHS,
        CommonFields.CURRENT_HOSPITALIZED,
        CommonFields.CURRENT_ICU,
        CommonFields.CURRENT_VENTILATED,
        CommonFields.CUMULATIVE_HOSPITALIZED,
        CommonFields.CUMULATIVE_ICU,
        CommonFields.POSITIVE_TESTS,
        CommonFields.NEGATIVE_TESTS,
        CommonFields.POSITIVE_TESTS_VIRAL,
        CommonFields.POSITIVE_CASES_VIRAL,
        CommonFields.TOTAL_TESTS,
        CommonFields.TOTAL_TESTS_VIRAL,
        CommonFields.TOTAL_TESTS_PEOPLE_VIRAL,
        CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL,
    ]
