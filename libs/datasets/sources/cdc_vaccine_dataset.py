from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class CDCVaccinesDataset(data_source.DataSource):
    SOURCE_NAME = "CDCVaccine"

    COMMON_DF_CSV_PATH = "data/vaccines-cdc/timeseries-common.csv"

    EXPECTED_FIELDS = [
        CommonFields.VACCINES_ALLOCATED,
        CommonFields.VACCINES_DISTRIBUTED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
    ]
