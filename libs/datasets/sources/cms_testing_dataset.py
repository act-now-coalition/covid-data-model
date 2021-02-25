from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class CMSTestingDataset(data_source.DataSource):
    SOURCE_TYPE = "CMSTesting"
    SOURCE_NAME = "The Centers for Medicare & Medicaid Services"
    SOURCE_URL = "https://data.cms.gov/stories/s/COVID-19-Nursing-Home-Data/bkwz-xpvg"

    COMMON_DF_CSV_PATH = "data/testing-cms/timeseries-common.csv"

    EXPECTED_FIELDS = [CommonFields.TEST_POSITIVITY_14D]
