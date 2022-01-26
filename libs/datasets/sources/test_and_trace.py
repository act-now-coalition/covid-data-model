from datapublic.common_fields import CommonFields
from libs.datasets import data_source


class TestAndTraceData(data_source.DataSource):
    SOURCE_TYPE = "TestAndTrace"

    COMMON_DF_CSV_PATH = "data/test-and-trace/state_data.csv"

    EXPECTED_FIELDS = [
        CommonFields.CONTACT_TRACERS_COUNT,
    ]
