from scripts import update_cdc_test_data

from libs.datasets import data_source


class CDCTestingDataset(data_source.CanScraperBase):
    SOURCE_NAME = "CDCTesting"

    TRANSFORM_METHOD = update_cdc_test_data.transform
