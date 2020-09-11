from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils


class NYTimesDataset(data_source.DataSource):
    SOURCE_NAME = "NYTimes"

    DATA_PATH = "data/cases-nytimes/timeseries-common.csv"

    HAS_AGGREGATED_NYC_BOROUGH = True

    INDEX_FIELD_MAP = {
        CommonFields.DATE: CommonFields.DATE,
        CommonFields.AGGREGATE_LEVEL: CommonFields.AGGREGATE_LEVEL,
        CommonFields.COUNTRY: CommonFields.COUNTRY,
        CommonFields.STATE: CommonFields.STATE,
        CommonFields.FIPS: CommonFields.FIPS,
    }

    COMMON_FIELD_MAP = {f: f for f in {CommonFields.CASES, CommonFields.DEATHS,}}

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        return cls(cls._rename_to_common_fields(data))
