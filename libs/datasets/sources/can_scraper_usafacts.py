from datapublic.common_fields import CommonFields

from libs.datasets.sources import can_scraper_helpers as ccd_helpers

from libs.datasets import data_source


class CANScraperUSAFactsProvider(data_source.CanScraperBase):
    SOURCE_TYPE = "USAFacts"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="cases",
            measurement="cumulative",
            unit="people",
            provider="usafacts",
            common_field=CommonFields.CASES,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="deaths",
            measurement="cumulative",
            unit="people",
            provider="usafacts",
            common_field=CommonFields.DEATHS,
        ),
    ]
