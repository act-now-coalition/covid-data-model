import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields


class TestAndTraceData(data_source.DataSource):
    DATA_PATH = "data/test-and-trace/state_data.csv"
    SOURCE_NAME = "TestAndTrace"

    class Fields(object):
        FIPS = CommonFields.FIPS
        STATE = CommonFields.STATE
        DATE = CommonFields.DATE
        CONTACT_TRACERS = CommonFields.CONTACT_TRACERS_COUNT

        # Not in the source file, added to conform to the expectations of this repo
        AGGREGATE_LEVEL = "aggregate_level"
        COUNTRY = "country"

    INDEX_FIELD_MAP = {
        CommonIndexFields.COUNTRY: Fields.COUNTRY,
        CommonIndexFields.STATE: Fields.STATE,
        CommonIndexFields.FIPS: Fields.FIPS,
        CommonIndexFields.DATE: Fields.DATE,
        CommonIndexFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.CONTACT_TRACERS_COUNT: Fields.CONTACT_TRACERS,
    }

    @classmethod
    def standardize_data(cls, data):
        data[cls.Fields.COUNTRY] = "USA"
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.STATE.value
        return data

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(input_path, dtype={cls.Fields.FIPS: str})
        data = cls.standardize_data(data)
        return cls(data)
