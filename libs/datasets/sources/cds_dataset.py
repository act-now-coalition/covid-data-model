from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets import dataset_utils


class CDSDataset(data_source.DataSource):
    DATA_PATH = "data/cases-cds/timeseries-common.csv"
    SOURCE_NAME = "CDS"

    INDEX_FIELD_MAP = {}

    # Keep in sync with update_cmdc.py in the covid-data-public repo.
    # DataSource objects must have a map from CommonFields to fields in the source file. For CMDC the
    # conversion is done in the covid-data-public repo so the map here doesn't represent any field renaming.
    COMMON_FIELD_MAP = {
        f: f
        for f in {
            CommonFields.DATE,
            CommonFields.COUNTRY,
            CommonFields.STATE,
            CommonFields.FIPS,
            CommonFields.AGGREGATE_LEVEL,
            CommonFields.CASES,
            CommonFields.DEATHS,
            CommonFields.POPULATION,
            CommonFields.NEGATIVE_TESTS,
            CommonFields.CUMULATIVE_HOSPITALIZED,
            CommonFields.CUMULATIVE_ICU,
        }
    }

    @classmethod
    def local(cls) -> "CDSDataset":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / cls.DATA_PATH
        return cls(common_df.read_csv(input_path))
