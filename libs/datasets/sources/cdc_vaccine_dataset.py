from libs.datasets import data_source
from datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


class CDCVaccinesDataset(data_source.CanScraperBase):
    SOURCE_TYPE = "CDCVaccine"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_allocated",
            measurement="cumulative",
            unit="doses",
            provider="cdc",
            common_field=CommonFields.VACCINES_ALLOCATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_distributed",
            measurement="cumulative",
            unit="doses",
            provider="cdc",
            common_field=CommonFields.VACCINES_DISTRIBUTED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_doses_administered",
            measurement="cumulative",
            unit="doses",
            provider="cdc",
            common_field=CommonFields.VACCINES_ADMINISTERED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_initiated",
            measurement="cumulative",
            unit="people",
            provider="cdc",
            common_field=CommonFields.VACCINATIONS_INITIATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_completed",
            measurement="cumulative",
            unit="people",
            provider="cdc",
            common_field=CommonFields.VACCINATIONS_COMPLETED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_additional_dose",
            measurement="cumulative",
            unit="people",
            provider="cdc",
            common_field=CommonFields.VACCINATIONS_ADDITIONAL_DOSE,
        ),
    ]
