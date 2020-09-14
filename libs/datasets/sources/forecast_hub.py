from covidactnow.datapublic import common_df
from libs.datasets import data_source
from libs.datasets import dataset_utils


class ForecastHubDataset(data_source.DataSource):
    SOURCE_NAME = "ForecastHub"

    DATA_PATH = "data/forecast-hub/timeseries-common.csv"

    INDEX_FIELD_MAP = {
        # Not Yet Implemented -> Currently only a move from covid-data-public to covid-data-model
    }

    COMMON_FIELD_MAP = {
        # Not Yet Implemented -> Currently only a move from covid-data-public to covid-data-model
    }

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        return cls(cls._rename_to_common_fields(data))
