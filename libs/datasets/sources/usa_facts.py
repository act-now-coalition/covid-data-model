from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class UsaFactsDataSource(data_source.DataSource):
    SOURCE_NAME = "USAFacts"

    COMMON_DF_CSV_PATH = "data/cases-covid-county-data/timeseries-usafacts.csv"

    COMMON_FIELD_MAP = {f: f for f in {CommonFields.CASES, CommonFields.DEATHS,}}

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        return cls(data)
