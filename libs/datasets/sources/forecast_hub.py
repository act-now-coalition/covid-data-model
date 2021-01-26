from functools import lru_cache

from covidactnow.datapublic import common_df
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries


class ForecastHubDataset(data_source.DataSource):
    SOURCE_NAME = "ForecastHub"

    DATA_PATH = "data/forecast-hub/timeseries-common.csv"

    COMMON_FIELD_MAP = {
        # Not Yet Implemented -> Currently only a move from covid-data-public to covid-data-model
    }

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        """
        This currently returns an empty dataset because _rename_to_common_fields restricts
        output to only columns found in INDEX_FIELD_MAP and COMMON_FIELD_MAP. This is not yet
        implemented because it is not required for this dataset to be merged via the combined
        dataset pathway. Specifically, which quantiles to persist has not finalized and as such
        they are not included in CommonFields and would be filtered out regardless.
        """
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path, set_index=False)
        return cls.make_timeseries_dataset(data)
