from covidactnow.datapublic.common_fields import CommonFields
from scripts import update_cdc_vaccine_data

from libs.datasets import data_source


class CDCVaccinesDataset(data_source.CanScraperBase):
    SOURCE_NAME = "CDCVaccine"

    TRANSFORM_METHOD = update_cdc_vaccine_data.transform

    EXPECTED_FIELDS = [
        CommonFields.VACCINES_ALLOCATED,
        CommonFields.VACCINES_DISTRIBUTED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
    ]
