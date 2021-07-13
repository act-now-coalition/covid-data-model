from libs.datasets import data_source
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


class CDCVaccinesDataset2(data_source.CanScraperBase):
    SOURCE_TYPE = "CDCVaccine2"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_allocated",
            measurement="cumulative",
            unit="doses",
            provider="cdc2",
            common_field=CommonFields.VACCINES_ALLOCATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_distributed",
            measurement="cumulative",
            unit="doses",
            provider="cdc2",
            common_field=CommonFields.VACCINES_DISTRIBUTED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_doses_administered",
            measurement="cumulative",
            unit="doses",
            provider="cdc2",
            common_field=CommonFields.VACCINES_ADMINISTERED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_initiated",
            measurement="cumulative",
            unit="people",
            provider="cdc2",
            common_field=CommonFields.VACCINATIONS_INITIATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_completed",
            measurement="cumulative",
            unit="people",
            provider="cdc2",
            common_field=CommonFields.VACCINATIONS_COMPLETED,
        ),
    ]
