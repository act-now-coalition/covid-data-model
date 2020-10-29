from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets.timeseries import TimeseriesDataset


class ValorumUsaFactsDataSource(data_source.DataSource):
    DATA_PATH = "data/cases-covid-county-data/timeseries-usafacts.csv"
    SOURCE_NAME = "ValorumUsaFacts"

    INDEX_FIELD_MAP = {f: f for f in TimeseriesDataset.INDEX_FIELDS}

    # Keep in sync with update_covid_county_data.py in the covid-data-public repo.
    # DataSource objects must have a map from CommonFields to fields in the source file.
    # For this source the conversion is done in the covid-data-public repo so the map
    # here doesn't represent any field renaming.
    COMMON_FIELD_MAP = {f: f for f in {CommonFields.CASES, CommonFields.DEATHS,}}

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        data = common_df.read_csv(input_path).reset_index()
        # Column names are already CommonFields so don't need to rename
        return cls(data, provenance=None)
