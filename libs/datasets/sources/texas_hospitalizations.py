from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class TexasHospitalizations(data_source.DataSource):
    SOURCE_NAME = "tx_hosp"

    COMMON_DF_CSV_PATH = "data/states/tx/tx_fips_hospitalizations.csv"

    EXPECTED_FIELDS = [CommonFields.CURRENT_HOSPITALIZED, CommonFields.CURRENT_ICU]
