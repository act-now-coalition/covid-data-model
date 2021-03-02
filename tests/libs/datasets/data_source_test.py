import pathlib

import more_itertools
import pytest
from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import data_source
from libs.datasets import taglib
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources import can_scraper_state_providers
from libs.datasets.sources import nytimes_dataset
from libs.datasets.sources import can_scraper_usafacts
from unittest import mock

from tests import test_helpers
from libs.datasets.taglib import UrlStr
from tests.libs.datasets.sources import can_scraper_helpers_test
from tests.libs.datasets.sources.can_scraper_helpers_test import build_can_scraper_dataframe


@pytest.mark.slow
def test_state_providers_smoke_test():
    """Make sure *something* is returned without any raised exceptions."""
    assert can_scraper_state_providers.CANScraperStateProviders.make_dataset()
    assert can_scraper_state_providers.CANScraperStateProviders.make_dataset()
    assert can_scraper_state_providers.CANScraperStateProviders.make_dataset.cache_info().hits > 0


@pytest.mark.parametrize("reverse_observation_order", [False, True])
def test_can_scraper_returns_source_url(reverse_observation_order):
    """Injects a tiny bit of data with a source_url in a CanScraperLoader."""

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

    class CANScraperForTest(can_scraper_usafacts.CANScraperUSAFactsProvider):
        @staticmethod
        def _get_covid_county_dataset():
            return ccd_helpers.CanScraperLoader(input_data)

    ds = CANScraperForTest.make_dataset()

    # Check that the URL gets all the way to the OneRegionTimeseriesDataset.
    one_region = ds.get_one_region(
        pipeline.Region.from_fips(can_scraper_helpers_test.DEFAULT_LOCATION)
    )
    assert one_region.sources(CommonFields.CASES) == [
        taglib.Source(type="USAFacts", url=UrlStr(more_itertools.last(test_url)))
    ]


def test_data_source_make_dataset(tmpdir):
    # Make a new subclass to keep this test separate from others in the make_dataset lru_cache.
    class NYTimesForTest(nytimes_dataset.NYTimesDataset):
        pass

    region = pipeline.Region.from_state("AZ")
    region_static = {CommonFields.STATE: "AZ", CommonFields.FIPS: "04"}
    tmp_data_root = pathlib.Path(tmpdir)
    csv_path = tmp_data_root / NYTimesForTest.COMMON_DF_CSV_PATH
    csv_path.parent.mkdir(parents=True)
    cases_ts = test_helpers.TimeseriesLiteral([10, 20, 30], source=NYTimesForTest.source_tag())
    deaths_ts = test_helpers.TimeseriesLiteral([1, 2, 3], source=NYTimesForTest.source_tag())

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
        dataset_read = NYTimesForTest.make_dataset()

    # This dataset is exactly like dataset_start except the timeseries include `source`. The test
    # builds it from scratch instead of calling dataset_start.add_tag_all so this test can find
    # problems with add_tag_all.
    dataset_expected = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: cases_ts, CommonFields.DEATHS: deaths_ts},
        region=region,
        static=region_static,
    )
    test_helpers.assert_dataset_like(dataset_expected, dataset_read)


def test_can_scraper_class_single_provider():
    cls: data_source.CanScraperBase
    for cls in test_helpers.get_concrete_subclasses(data_source.CanScraperBase):
        providers = set(v.provider for v in cls.VARIABLES)
        assert len(providers) == 1


def test_can_scraper_class_unique_variable_names():
    cls: data_source.CanScraperBase
    for cls in test_helpers.get_concrete_subclasses(data_source.CanScraperBase):
        variables_not_none = [v for v in cls.VARIABLES if v.common_field is not None]
        # Source `variable_name` may be duplicated with different measurement or unit.
        assert len(variables_not_none) == len(
            set((v.variable_name, v.measurement, v.unit) for v in variables_not_none)
        )
        # Destination `common_field` is expected to be unique
        assert len(variables_not_none) == len(set(v.common_field for v in variables_not_none))


def test_can_scraper_class_variable_set_expected_fields_unset():
    cls: data_source.CanScraperBase
    for cls in test_helpers.get_concrete_subclasses(data_source.CanScraperBase):
        # EXPECTED_FIELDS is empty. It is computed from VARIABLES.
        assert not cls.EXPECTED_FIELDS
        assert cls.VARIABLES
