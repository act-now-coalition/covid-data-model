from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset


class CMSTestingDataset(data_source.DataSource):
    SOURCE_NAME = "CMSTesting"
    FILL_MISSING_STATE_LEVEL_DATA = False

    DATA_PATH = "data/testing-cms/timeseries-common.csv"

    INDEX_FIELD_MAP = {f: f for f in TimeseriesDataset.INDEX_FIELDS}

    COMMON_FIELD_MAP = {f: f for f in {CommonFields.TEST_POSITIVITY_14D}}

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path, set_index=False)
        return cls(data)
