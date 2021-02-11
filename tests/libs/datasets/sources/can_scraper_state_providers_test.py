from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources import can_scraper_state_providers
from unittest import mock

from libs.datasets.sources.can_scraper_usafacts import CANScraperUSAFactsProvider
from tests.libs.datasets.sources import can_scraper_helpers_test
from tests.libs.datasets.sources.can_scraper_helpers_test import _build_can_scraper_dataframe


def test_state_providers_smoke_test():
    can_scraper_state_providers.CANScraperStateProviders.make_dataset()


def test_state_providers():
    variable = ccd_helpers.ScraperVariable(
        variable_name="cases",
        measurement="cumulative",
        unit="people",
        provider="usafacts",
        common_field=CommonFields.CASES,
    )
    test_url = "http://foo.com"

    input_data = _build_can_scraper_dataframe({variable: [10, 20, 30]}, source_url=test_url,)
    data = ccd_helpers.CanScraperLoader(input_data)

    with mock.patch("libs.datasets.data_source.CanScraperBase") as mock_can_scraper_base:
        mock_can_scraper_base._get_covid_county_dataset.return_value = data
        ds = CANScraperUSAFactsProvider.make_dataset()
    one_region = ds.get_one_region(
        pipeline.Region.from_fips(can_scraper_helpers_test.DEFAULT_LOCATION)
    )
    assert one_region.source_url == {CommonFields.CASES: [test_url]}
