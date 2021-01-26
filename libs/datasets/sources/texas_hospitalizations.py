from functools import lru_cache

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
from libs.datasets import data_source
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS


class TexasHospitalizations(data_source.DataSource):
    COMMON_DF_CSV_PATH = "data/states/tx/tx_fips_hospitalizations.csv"
    SOURCE_NAME = "tx_hosp"

    EXPECTED_FIELDS = [CommonFields.CURRENT_HOSPITALIZED, CommonFields.CURRENT_ICU]
