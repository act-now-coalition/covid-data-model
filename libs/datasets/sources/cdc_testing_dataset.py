from covidactnow.datapublic.common_fields import CommonFields

# TODO(tom): Remove this really ugly import from the covid-data-public repo.
from scripts import update_cdc_test_data

from libs.datasets import data_source


class CDCTestingDataset(data_source.CanScraperBase):
    SOURCE_NAME = "CDCTesting"

    TRANSFORM_METHOD = update_cdc_test_data.transform

    EXPECTED_FIELDS = [
        CommonFields.TEST_POSITIVITY_7D,
    ]
