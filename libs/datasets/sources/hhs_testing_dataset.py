from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class HHSTestingDataset(data_source.DataSource):
    SOURCE_TYPE = "HHSTesting"

    COMMON_DF_CSV_PATH = "data/testing-hhs/timeseries-common.csv"

    EXPECTED_FIELDS = [CommonFields.NEGATIVE_TESTS, CommonFields.POSITIVE_TESTS]
