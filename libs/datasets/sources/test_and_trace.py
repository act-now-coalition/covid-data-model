import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


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

    INDEX_FIELD_MAP = {f: f for f in TIMESERIES_INDEX_FIELDS}

    COMMON_FIELD_MAP = {f: f for f in [CommonFields.CONTACT_TRACERS_COUNT]}

    @classmethod
    def standardize_data(cls, data):
        data[cls.Fields.COUNTRY] = "USA"
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.STATE.value
        data = cls._rename_to_common_fields(data)
        return data

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = pd.read_csv(input_path, parse_dates=[cls.Fields.DATE], dtype={cls.Fields.FIPS: str})
        data = cls.standardize_data(data)
        return cls(data)
