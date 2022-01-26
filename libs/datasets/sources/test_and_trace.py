from datapublic.common_fields import CommonFields
from libs.datasets import data_source


class TestAndTraceData(data_source.DataSource):
    SOURCE_TYPE = "TestAndTrace"

    # 1/26/2022: This data is no longer updating. We keep it in the
    # data/ directory so that this dataset builds successfully, but
    # nothing is included in the API because the data is (very) stale.
    COMMON_DF_CSV_PATH = "misc/stale_test_and_trace_data.csv"

    EXPECTED_FIELDS = [
        CommonFields.CONTACT_TRACERS_COUNT,
    ]
