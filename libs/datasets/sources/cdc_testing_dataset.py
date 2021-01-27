from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class CDCTestingDataset(data_source.DataSource):
    SOURCE_NAME = "CDCTesting"

    COMMON_DF_CSV_PATH = "data/testing-cdc/timeseries-common.csv"

    INDEX_FIELD_MAP = {f: f for f in TIMESERIES_INDEX_FIELDS}

    COMMON_FIELD_MAP = {f: f for f in {CommonFields.TEST_POSITIVITY_7D}}

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path, set_index=False)
        return cls(data)
