from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


class NYTimesDataset(data_source.CanScraperBase):
    SOURCE_TYPE = "NYTimes"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="cases",
            measurement="cumulative",
            unit="people",
            provider="nyt",
            common_field=CommonFields.CASES,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="deaths",
            measurement="cumulative",
            unit="people",
            provider="nyt",
            common_field=CommonFields.DEATHS,
        ),
    ]
