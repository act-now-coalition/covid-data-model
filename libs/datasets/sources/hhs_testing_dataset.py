from datapublic.common_fields import CommonFields
from libs.datasets import data_source
from libs.datasets.sources import can_scraper_helpers


class HHSTestingDataset(data_source.CanScraperBase):
    SOURCE_TYPE = "HHSTesting"

    VARIABLES = [
        can_scraper_helpers.ScraperVariable(
            variable_name="pcr_tests_negative",
            measurement="cumulative",
            unit="specimens",
            provider="hhs",
            common_field=CommonFields.NEGATIVE_TESTS,
        ),
        can_scraper_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="cumulative",
            unit="specimens",
            provider="hhs",
            common_field=CommonFields.POSITIVE_TESTS,
        ),
    ]
