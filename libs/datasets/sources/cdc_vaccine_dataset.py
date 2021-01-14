from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


class CDCVaccinesDataset(data_source.DataSource):
    SOURCE_NAME = "CDCVaccine"

    DATA_PATH = "data/vaccines-cdc/timeseries-common.csv"

    INDEX_FIELD_MAP = {f: f for f in TIMESERIES_INDEX_FIELDS}

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
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path, set_index=False)
        return cls(data)
