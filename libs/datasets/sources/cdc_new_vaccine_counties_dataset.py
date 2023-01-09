from libs.datasets import data_source
from datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


class CDCNewVaccinesCountiesDataset(data_source.CanScraperBase):
    """In ~June 2021, CDC added a new source for county vaccine data including
    1st dose data and historical timeseries data. This is surfaced by the
    scrapers as provider=cdc2 and we now prefer it over the old data source."""

    SOURCE_TYPE = "CDCNewVaccinesCountiesDataset"

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
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_additional_dose",
            measurement="cumulative",
            unit="people",
            provider="cdc2",
            common_field=CommonFields.VACCINATIONS_ADDITIONAL_DOSE,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_additional_dose",  # TODO: USE BIVALENT DATA ONCE IT'S IN THE PARQUET
            measurement="cumulative",
            unit="people",
            provider="cdc2",
            common_field=CommonFields.VACCINATIONS_BIVALENT_DOSE,
        ),
    ]
