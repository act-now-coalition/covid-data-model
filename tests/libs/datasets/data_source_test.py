import more_itertools
import pytest
from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import data_source
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources import can_scraper_state_providers
from unittest import mock

from libs.datasets.sources.can_scraper_usafacts import CANScraperUSAFactsProvider
from tests.libs.datasets.sources import can_scraper_helpers_test
from tests.libs.datasets.sources.can_scraper_helpers_test import build_can_scraper_dataframe


@pytest.mark.slow
def test_state_providers_smoke_test():
    """Make sure *something* is returned without any raised exceptions."""
    assert can_scraper_state_providers.CANScraperStateProviders.make_dataset()


@pytest.mark.parametrize("reverse_observation_order", [False, True])
def test_add_source_url_to_dataset(reverse_observation_order):
    variable = ccd_helpers.ScraperVariable(
        variable_name="cases",
        measurement="cumulative",
        unit="people",
        provider="usafacts",
        common_field=CommonFields.CASES,
    )
    test_url = [f"http://foo.com/{i}" for i in range(3)]
    input_data = build_can_scraper_dataframe({variable: [10, 20, 30]}, source_url=test_url)

    if reverse_observation_order:
        input_data = input_data.iloc[::-1]

    data = ccd_helpers.CanScraperLoader(input_data)

    with mock.patch("libs.datasets.data_source.CanScraperBase") as mock_can_scraper_base:
        # Clear lru_cache so values of previous tests are not used.
        CANScraperUSAFactsProvider.make_dataset.cache_clear()
        mock_can_scraper_base._get_covid_county_dataset.return_value = data
        ds = CANScraperUSAFactsProvider.make_dataset()
        # Clear lru_cache so subsequent tests don't get the mocked data.
        CANScraperUSAFactsProvider.make_dataset.cache_clear()

    one_region = ds.get_one_region(
        pipeline.Region.from_fips(can_scraper_helpers_test.DEFAULT_LOCATION)
    )
    assert one_region.source_url == {CommonFields.CASES: [more_itertools.last(test_url)]}
