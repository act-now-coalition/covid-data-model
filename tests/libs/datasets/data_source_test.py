import pathlib

import more_itertools
import pytest
import pandas as pd
import structlog
from datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import data_source
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources import can_scraper_local_dashboard_providers
from libs.datasets.sources import nytimes_dataset
from libs.datasets.sources import can_scraper_usafacts
from unittest import mock

from libs.pipeline import Region
from tests import test_helpers
from libs.datasets.taglib import UrlStr
from tests.libs.datasets.sources import can_scraper_helpers_test
from tests.libs.datasets.sources.can_scraper_helpers_test import build_can_scraper_dataframe
from tests.test_helpers import TimeseriesLiteral


@pytest.mark.slow
@pytest.mark.skip(
    "https://trello.com/c/FFAUKu3k/ - This is crashing in CI (probably running out of memory)."
)
def test_state_providers_smoke_test():
    """Make sure *something* is returned without any raised exceptions."""
    assert can_scraper_local_dashboard_providers.CANScraperStateProviders.make_dataset()
    assert can_scraper_local_dashboard_providers.CANScraperStateProviders.make_dataset()
    assert (
        can_scraper_local_dashboard_providers.CANScraperStateProviders.make_dataset.cache_info().hits
        > 0
    )


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
        pipeline.Region.from_location_id(can_scraper_helpers_test.DEFAULT_LOCATION_ID)
    )
    assert one_region.sources_all_bucket(CommonFields.CASES) == [
        taglib.Source(type="USAFacts", url=UrlStr(more_itertools.last(test_url)))
    ]


@pytest.mark.skip(
    reason="01/12/2022: NYTimesDataset now reads from the Parquet file, rendering this obsolete."
)
def test_data_source_make_dataset(tmpdir):
    # Make a new subclass to keep this test separate from others in the make_dataset lru_cache.
    class NYTimesForTest(nytimes_dataset.NYTimesDataset):
        pass

    region = pipeline.Region.from_state("AZ")
    tmp_data_root = pathlib.Path(tmpdir)
    csv_path = tmp_data_root / NYTimesForTest.COMMON_DF_CSV_PATH
    csv_path.parent.mkdir(parents=True)
    cases_ts = test_helpers.TimeseriesLiteral([10, 20, 30], source=NYTimesForTest.source_tag())
    deaths_ts = test_helpers.TimeseriesLiteral([1, 2, 3], source=NYTimesForTest.source_tag())

    # Make a tiny fake NYTimes dataset and write it to disk.
    dataset_start = test_helpers.build_default_region_dataset(
        # Make timeseries using just the real values, no source.
        {CommonFields.CASES: cases_ts.data, CommonFields.DEATHS: deaths_ts.data},
        region=region,
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
        {CommonFields.CASES: cases_ts, CommonFields.DEATHS: deaths_ts}, region=region,
    )
    test_helpers.assert_dataset_like(dataset_expected, dataset_read)


@pytest.mark.skip(
    reason=(
        "01/12/2022: NYTimesDataset now reads from the Parquet file"
        "making this test redundant to test_data_source_truncates_dates_can_scraper"
    )
)
def test_data_source_truncates_dates():
    df = test_helpers.read_csv_str(
        "fips,      date,cases,deaths\n"
        "  01,2019-12-31,  100,     1\n"
        "  01,2020-01-01,  200,     2\n",
        skip_spaces=True,
    )

    class NYTimesForTest(nytimes_dataset.NYTimesDataset):
        @classmethod
        def _load_data(cls) -> pd.DataFrame:
            return df

    with structlog.testing.capture_logs() as logs:
        ds = NYTimesForTest.make_dataset()

    assert ds.timeseries_bucketed_wide_dates.columns.to_list() == pd.to_datetime(["2020-01-01"])
    assert more_itertools.one(logs)["event"] == "Dropping old data"


def test_data_source_bad_fips():
    df = test_helpers.read_csv_str(
        "fips,                date,       cases\n"
        "06010,         2020-04-01,         100\n"
        "17031,         2020-04-01,         100\n",
        skip_spaces=True,
    )

    class DataSourceForTest(data_source.DataSource):
        EXPECTED_FIELDS = [CommonFields.CASES]
        SOURCE_TYPE = "TestSource"
        SOURCE_NAME = "Test source"
        SOURCE_URL = "http://public.test.gov"

        @classmethod
        def _load_data(cls) -> pd.DataFrame:
            return df

    with structlog.testing.capture_logs() as logs:
        ds = DataSourceForTest.make_dataset()
    assert more_itertools.one(logs)["event"] == timeseries.NO_LOCATION_ID_FOR_FIPS

    ds_expected = test_helpers.build_dataset(
        {
            Region.from_fips("17031"): {
                CommonFields.CASES: TimeseriesLiteral([100], source=DataSourceForTest.source_tag())
            }
        }
    )
    test_helpers.assert_dataset_like(ds, ds_expected)


def test_data_source_truncates_dates_can_scraper():
    """A second test for data truncation. See comment at top of _check_data."""
    variable_cases = ccd_helpers.ScraperVariable(
        variable_name="cases",
        measurement="cumulative",
        unit="people",
        provider="usafacts",
        common_field=CommonFields.CASES,
    )
    variable_deaths = ccd_helpers.ScraperVariable(
        variable_name="deaths",
        measurement="cumulative",
        unit="people",
        provider="usafacts",
        common_field=CommonFields.DEATHS,
    )
    input_data = build_can_scraper_dataframe(
        {variable_cases: [10, 20], variable_deaths: [1, 2]}, start_date="2019-12-31"
    )

    class CANScraperForTest(can_scraper_usafacts.CANScraperUSAFactsProvider):
        @staticmethod
        def _get_covid_county_dataset():
            return ccd_helpers.CanScraperLoader(input_data)

    with structlog.testing.capture_logs() as logs:
        ds = CANScraperForTest.make_dataset()

    assert ds.timeseries_bucketed_wide_dates.columns.to_list() == pd.to_datetime(["2020-01-01"])
    assert more_itertools.one(logs)["event"] == "Dropping old data"


def test_can_scraper_class_single_provider():
    cls: data_source.CanScraperBase
    for cls in test_helpers.get_concrete_subclasses_not_in_tests(data_source.CanScraperBase):
        providers = set(v.provider for v in cls.VARIABLES)
        assert len(providers) == 1


def test_can_scraper_class_unique_variable_names():
    cls: data_source.CanScraperBase
    for cls in test_helpers.get_concrete_subclasses_not_in_tests(data_source.CanScraperBase):
        variables_not_none = [v for v in cls.VARIABLES if v.common_field is not None]
        # Source `variable_name` may be duplicated with different measurement or unit.
        assert len(variables_not_none) == len(
            set((v.variable_name, v.measurement, v.unit) for v in variables_not_none)
        )
        # Destination `common_field` is expected to be unique
        assert len(variables_not_none) == len(set(v.common_field for v in variables_not_none))


def test_can_scraper_class_variable_set():
    cls: data_source.CanScraperBase
    for cls in test_helpers.get_concrete_subclasses_not_in_tests(data_source.CanScraperBase):
        assert cls.VARIABLES


def test_can_scraper_class_expected_fields_set():
    # EXPECTED_FIELD is not set in the base class but is set in all concrete subclasses,
    # though it may be an empty list.
    with pytest.raises(AttributeError):
        data_source.DataSource.EXPECTED_FIELDS
    cls: data_source.DataSource
    for cls in test_helpers.get_concrete_subclasses_not_in_tests(data_source.DataSource):
        assert isinstance(cls.EXPECTED_FIELDS, list)
        assert cls.SOURCE_TYPE
        assert isinstance(cls.SOURCE_TYPE, str)
