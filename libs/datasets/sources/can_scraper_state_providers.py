from covidactnow.datapublic.common_fields import CommonFields

# TODO(tom): Remove this really ugly import from the covid-data-public repo.
from scripts import update_can_scraper_state_providers

from libs.datasets import data_source


class CANScraperStateProviders(data_source.CanScraperBase):
    SOURCE_NAME = "CANScrapersStateProviders"

    TRANSFORM_METHOD = update_can_scraper_state_providers.transform

    EXPECTED_FIELDS = [
        CommonFields.STAFFED_BEDS,
        CommonFields.CASES,
        CommonFields.DEATHS,
        CommonFields.VACCINES_ALLOCATED,
        CommonFields.VACCINES_ADMINISTERED,
        CommonFields.VACCINES_DISTRIBUTED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
        CommonFields.TOTAL_TESTS_VIRAL,
        CommonFields.ICU_BEDS,
        CommonFields.CURRENT_HOSPITALIZED,
        CommonFields.POSITIVE_TESTS_VIRAL,
        CommonFields.CURRENT_ICU,
        CommonFields.VACCINATIONS_INITIATED_PCT,
        CommonFields.VACCINATIONS_COMPLETED_PCT,
    ]
