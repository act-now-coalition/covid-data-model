from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class NYTimesDataset(data_source.DataSource):
    SOURCE_TYPE = "NYTimes"

    COMMON_DF_CSV_PATH = "data/cases-nytimes/timeseries-common.csv"

    EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]

    IGNORED_FIELDS = data_source.DataSource.IGNORED_FIELDS + (CommonFields.STATE_FULL_NAME,)
