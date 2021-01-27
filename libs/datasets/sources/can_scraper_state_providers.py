from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source


class CANScraperStateProviders(data_source.DataSource):
    SOURCE_NAME = "CANScrapersStateProviders"

    COMMON_DF_CSV_PATH = "data/can-scrapers-state-providers/timeseries-common.csv"

    EXPECTED_FIELDS = [
        CommonFields.VACCINES_ALLOCATED,
        CommonFields.VACCINES_DISTRIBUTED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
        CommonFields.TOTAL_TESTS_VIRAL,
        CommonFields.ICU_BEDS,
        CommonFields.CURRENT_HOSPITALIZED,
        CommonFields.POSITIVE_TESTS_VIRAL,
        CommonFields.CURRENT_ICU,
    ]
