from functools import lru_cache

from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries


class CDCVaccinesDataset(data_source.DataSource):
    SOURCE_NAME = "CDCVaccine"

    DATA_PATH = "data/vaccines-cdc/timeseries-common.csv"

    # TODO: Remove constants(https://trello.com/c/uPQXBuDg/777-remove-unused-datasource-constants)
    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.VACCINES_ALLOCATED,
            CommonFields.VACCINES_DISTRIBUTED,
            CommonFields.VACCINATIONS_INITIATED,
            CommonFields.VACCINATIONS_COMPLETED,
        }
    }

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path, set_index=False)
        return cls.make_timeseries_dataset(data)
