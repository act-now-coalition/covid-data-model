import pathlib

import more_itertools
import pytest
from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import taglib
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources import can_scraper_state_providers
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from unittest import mock

from libs.datasets.sources.can_scraper_usafacts import CANScraperUSAFactsProvider
from tests import test_helpers
from tests.libs.datasets.sources import can_scraper_helpers_test
from tests.libs.datasets.sources.can_scraper_helpers_test import build_can_scraper_dataframe


@pytest.mark.slow
def test_state_providers_smoke_test():
    """Make sure *something* is returned without any raised exceptions."""
    assert can_scraper_state_providers.CANScraperStateProviders.make_dataset()


@pytest.mark.parametrize("reverse_observation_order", [False, True])
def test_can_scraper_usa_facts_provider_returns_source_url(reverse_observation_order):
    """Injects a tiny bit of data in a CanScraperLoader with a source_url for a quick test."""
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
        # Reversing the order to check that the source_url of the last date is returned when the
        # observations are not sorted by increasing date.
        input_data = input_data.iloc[::-1]

    data = ccd_helpers.CanScraperLoader(input_data)

    with mock.patch("libs.datasets.data_source.CanScraperBase") as mock_can_scraper_base:
        # Clear lru_cache so values of previous tests are not used.
        CANScraperUSAFactsProvider.make_dataset.cache_clear()
        mock_can_scraper_base._get_covid_county_dataset.return_value = data
        ds = CANScraperUSAFactsProvider.make_dataset()
        # Clear lru_cache so subsequent tests don't get the mocked data.
        CANScraperUSAFactsProvider.make_dataset.cache_clear()

    # Check that the URL gets all the way to the OneRegionTimeseriesDataset.
    one_region = ds.get_one_region(
        pipeline.Region.from_fips(can_scraper_helpers_test.DEFAULT_LOCATION)
    )
    assert one_region.source_url == {CommonFields.CASES: [more_itertools.last(test_url)]}


def test_data_source_make_dataset(tmpdir):
    region = pipeline.Region.from_state("AZ")
    region_static = {CommonFields.STATE: "AZ", CommonFields.FIPS: "04"}
    tmp_data_root = pathlib.Path(tmpdir)
    csv_path = tmp_data_root / NYTimesDataset.COMMON_DF_CSV_PATH
    csv_path.parent.mkdir(parents=True)
    cases_ts = test_helpers.TimeseriesLiteral([10, 20, 30], source=NYTimesDataset.source_tag())
    deaths_ts = test_helpers.TimeseriesLiteral([1, 2, 3], source=NYTimesDataset.source_tag())

    # Make a tiny fake NYTimes dataset and write it to disk.
    dataset_start = test_helpers.build_default_region_dataset(
        # Make timeseries using just the real values, no source.
        {CommonFields.CASES: list(cases_ts), CommonFields.DEATHS: list(deaths_ts)},
        region=region,
        static=region_static,
    )
    dataset_start.to_csv(csv_path, include_latest=False)

    # Load the fake data using the normal code path.
    with mock.patch("libs.datasets.data_source.dataset_utils") as mock_can_scraper_base:
        mock_can_scraper_base.LOCAL_PUBLIC_DATA_PATH = tmp_data_root
        dataset_read = NYTimesDataset.make_dataset()

    # This dataset is exactly like dataset_start except the timeseries include `source`. The test
    # builds it from scratch instead of calling dataset_start.add_tag_all so this test can find
    # problems with add_tag_all.
    dataset_expected = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: cases_ts, CommonFields.DEATHS: deaths_ts},
        region=region,
        static=region_static,
    )
    test_helpers.assert_dataset_like(dataset_expected, dataset_read)
