from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class CMSTestingDataset(data_source.DataSource):
    SOURCE_NAME = "CMSTesting"

    COMMON_DF_CSV_PATH = "data/testing-cms/timeseries-common.csv"

    EXPECTED_FIELDS = [CommonFields.TEST_POSITIVITY_14D]
