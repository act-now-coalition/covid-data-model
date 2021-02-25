from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets.taglib import UrlStr


class NYTimesDataset(data_source.DataSource):
    SOURCE_TYPE = "NYTimes"
    SOURCE_NAME = "The New York Times"
    SOURCE_URL = UrlStr("https://github.com/nytimes/covid-19-data")

    COMMON_DF_CSV_PATH = "data/cases-nytimes/timeseries-common.csv"

    EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]

    IGNORED_FIELDS = data_source.DataSource.IGNORED_FIELDS + (CommonFields.STATE_FULL_NAME,)
