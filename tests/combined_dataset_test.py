import logging
import re

import structlog
import pandas as pd

from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets, CommonFields
from libs.datasets import data_source
from libs.datasets import timeseries
from libs.datasets.combined_datasets import provenance_wide_metrics_to_series

from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.can_scraper_usafacts import CANScraperUSAFactsProvider

from libs.pipeline import Region
from libs.pipeline import RegionMask
from tests import test_helpers
from tests.dataset_utils_test import read_csv_and_index_fips_date
import pytest

# Tests to make sure that combined datasets are building data with unique indexes
# If this test is failing, it means that there is one of the data sources that
# is returning multiple values for a single row.
@pytest.mark.slow
def test_unique_index_values_us_timeseries():
    us_dataset = combined_datasets.load_us_timeseries_dataset()
    us_df = us_dataset.timeseries.reset_index()
    duplicates = us_df.duplicated([CommonFields.LOCATION_ID, CommonFields.DATE], keep=False)
    assert not duplicates.any(), us_df.loc[duplicates, :]


# Check some counties picked arbitrarily: San Francisco/06075 and Houston (Harris County, TX)/48201
@pytest.mark.slow
@pytest.mark.parametrize("fips", ["06075", "48201"])
def test_combined_county_has_some_data(fips):
    region_data = combined_datasets.load_us_timeseries_dataset().get_one_region(
        Region.from_fips(fips)
    )
    assert region_data.data[CommonFields.POSITIVE_TESTS].all()
    assert region_data.data[CommonFields.NEGATIVE_TESTS].all()
    assert region_data.latest[CommonFields.DEATHS] > 1


# Check some counties picked arbitrarily: (Orange County, CA)/06059 and (Harris County, TX)/48201
@pytest.mark.slow
@pytest.mark.parametrize("fips", ["06059", "48201"])
def test_combined_county_has_some_timeseries_data(fips):
    region = Region.from_fips(fips)
    latest = combined_datasets.load_us_timeseries_dataset().get_one_region(region)
    date = "2020-09-04"  # Arbitrary date when both regions have data
    df = latest.data.set_index(CommonFields.DATE)
    one_date = df.loc[date]
    assert one_date[CommonFields.CASES] > 0
    assert one_date[CommonFields.DEATHS] > 0
    # Check that there is some testing data, either positive and negative tests or a test
    # positivity ratio.
    assert (
        one_date[CommonFields.POSITIVE_TESTS] > 0 and one_date[CommonFields.NEGATIVE_TESTS] > 0
    ) or (
        one_date[CommonFields.TEST_POSITIVITY_7D] > 0
        or one_date[CommonFields.TEST_POSITIVITY_14D] > 0
    )
    assert one_date[CommonFields.CURRENT_ICU] > 0


@pytest.mark.slow
def test_get_county_name():
    assert combined_datasets.get_county_name(Region.from_fips("06059")) == "Orange County"
    assert combined_datasets.get_county_name(Region.from_fips("48201")) == "Harris County"


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_source_cls", [NYTimesDataset,],
)
@pytest.mark.skip(
    reason="01/12/2022: NYTimesDataset now reads from the Parquet file, reducing the usefulness of this test."
)
def test_unique_timeseries(data_source_cls):
    dataset = data_source_cls.make_dataset()
    # Check for duplicate rows with the same INDEX_FIELDS. Sort by index so duplicates are next to
    # each other in the message if the assert fails.
    timeseries_data = dataset.timeseries.sort_index()
    assert timeseries_data.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
    duplicates = timeseries_data.index.duplicated(keep=False)
    assert not duplicates.any(), str(timeseries_data.loc[duplicates])


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_source_cls", [NYTimesDataset],
)
@pytest.mark.skip(
    reason="01/12/2022: NYTimesDataset now reads from the Parquet file, reducing the usefulness of this test."
)
def test_expected_field_in_sources(data_source_cls):
    dataset = data_source_cls.make_dataset()

    assert not dataset.timeseries.empty
    assert not dataset.geo_data.empty

    states = set(dataset.geo_data[CommonFields.STATE])

    good_state = set()
    for state in states:
        if pd.notna(state) and re.fullmatch(r"[A-Z]{2}", state):
            good_state.add(state)
        else:
            logging.info(f"Ignoring {state} in {data_source_cls.__name__}")
    assert len(good_state) >= 48


def test_melt_provenance():
    wide = read_csv_and_index_fips_date(
        "fips,date,cases,recovered\n"
        "97111,2020-04-01,source_a,source_b\n"
        "97111,2020-04-02,source_a,\n"
        "97222,2020-04-01,source_c,\n"
    )
    with structlog.testing.capture_logs() as logs:
        long = provenance_wide_metrics_to_series(wide, structlog.get_logger())

    assert logs == []

    assert long.to_dict() == {
        ("97111", "cases"): "source_a",
        ("97111", "recovered"): "source_b",
        ("97222", "cases"): "source_c",
    }


def test_melt_provenance_multiple_sources():
    wide = read_csv_and_index_fips_date(
        "fips,date,cases,recovered\n"
        "97111,2020-04-01,source_a,source_b\n"
        "97111,2020-04-02,source_x,\n"
        "97222,2020-04-01,source_c,\n"
    )
    with structlog.testing.capture_logs() as logs:
        long = provenance_wide_metrics_to_series(wide, structlog.get_logger())

    assert [l["event"] for l in logs] == ["Multiple rows for a timeseries"]

    assert long.to_dict() == {
        ("97111", "cases"): "source_a;source_x",
        ("97111", "recovered"): "source_b",
        ("97222", "cases"): "source_c",
    }


def test_make_latest_from_timeseries_simple():
    data = read_csv_and_index_fips_date(
        "fips,county,state,country,date,aggregate_level,m1,m2\n"
        "97111,Smith County,ZZ,USA,2020-04-01,county,1,\n"
        "97111,Smith County,ZZ,USA,2020-04-02,county,,2\n"
    ).reset_index()
    ds = timeseries.MultiRegionDataset.from_fips_timeseries_df(data)
    region = ds.get_one_region(Region.from_fips("97111"))
    # Compare 2 values in region.latest
    expected = {"m1": 1, "m2": 2}
    actual = {key: region.latest[key] for key in expected.keys()}
    assert actual == expected


@pytest.mark.skip("county is not preserved. maybe warn when it differs from geo-data.csv.")
def test_make_latest_from_timeseries_dont_touch_county():
    data = read_csv_and_index_fips_date(
        "fips,county,state,country,date,aggregate_level,m1,m2\n"
        "95123,Smith Countyy,YY,USA,2020-04-01,county,1,\n"
        "97123,Smith Countzz,ZZ,USA,2020-04-01,county,2,\n"
        "56,,WY,USA,2020-04-01,state,3,\n"
    ).reset_index()
    ds = timeseries.MultiRegionDataset.from_fips_timeseries_df(data)

    def get_latest(region) -> dict:
        """Returns an interesting subset of latest for given region"""
        latest = ds.get_one_region(region).latest
        return {key: latest[key] for key in ["county", "m1", "m2"] if latest.get(key) is not None}

    assert get_latest(Region.from_fips("95123")) == {
        "m1": 1,
        "county": "Smith Countyy",
    }
    assert get_latest(Region.from_fips("97123")) == {
        "m1": 2,
        "county": "Smith Countzz",
    }
    assert get_latest(Region.from_state("WY")) == {"m1": 3}


def test_combined_datasets_uses_only_expected_fields():
    """Checks that field sources in combined_datasets appear in the corresponding EXPECTED_FIELDS"""
    for field_name, sources in combined_datasets.ALL_TIMESERIES_FEATURE_DEFINITION.items():
        for source in sources:
            assert field_name in source.EXPECTED_FIELDS, (
                f"{source.SOURCE_TYPE} is in combined_datasets for {field_name} but the field "
                f"is not in the {source.SOURCE_TYPE} EXPECTED_FIELDS."
            )


def test_dataclass_include_exclude_attributes():
    """Tests just the class attributes without calling the slow make_dataset()."""
    orig_data_source_cls = CANScraperUSAFactsProvider

    ny_source = combined_datasets.datasource_regions(
        orig_data_source_cls, RegionMask(states=["NY"])
    )
    # pylint: disable=E1101
    assert ny_source.SOURCE_TYPE == orig_data_source_cls.SOURCE_TYPE
    assert ny_source.EXPECTED_FIELDS == orig_data_source_cls.EXPECTED_FIELDS


def test_dataclass_include_exclude():
    """Tests datasource_regions using mock data for speed."""
    region_data = {CommonFields.CASES: [100, 200, 300], CommonFields.DEATHS: [0, 1, 2]}
    regions_orig = [Region.from_state(state) for state in "AZ CA NY IL TX".split()] + [
        Region.from_fips(fips) for fips in "06037 06045 17031 17201".split()
    ]
    dataset_orig = test_helpers.build_dataset({region: region_data for region in regions_orig})

    # Make a new subclass to keep this test separate from others in the make_dataset lru_cache.
    class DataSourceForTest(data_source.DataSource):
        EXPECTED_FIELDS = [CommonFields.CASES, CommonFields.DEATHS]
        SOURCE_TYPE = "DataSourceForTest"

        @classmethod
        def make_dataset(cls) -> timeseries.MultiRegionDataset:
            return dataset_orig

    orig_data_source_cls = DataSourceForTest
    orig_ds = orig_data_source_cls.make_dataset()
    assert "iso1:us#iso2:us-tx" in orig_ds.location_ids
    assert "iso1:us#iso2:us-ny" in orig_ds.location_ids

    ny_source = combined_datasets.datasource_regions(
        orig_data_source_cls, RegionMask(states=["NY"])
    )
    ny_ds = ny_source.make_dataset()
    assert "iso1:us#iso2:us-tx" not in ny_ds.location_ids
    assert "iso1:us#iso2:us-ny" in ny_ds.location_ids

    ca_counties_without_la_source = combined_datasets.datasource_regions(
        orig_data_source_cls,
        RegionMask(AggregationLevel.COUNTY, states=["CA"]),
        exclude=Region.from_fips("06037"),
    )
    ds = ca_counties_without_la_source.make_dataset()
    assert "iso1:us#iso2:us-tx" not in ds.location_ids
    assert "iso1:us#iso2:us-ca" not in ds.location_ids
    assert "iso1:us#iso2:us-ca#fips:06045" in ds.location_ids
    assert "iso1:us#iso2:us-ca#fips:06037" not in ds.location_ids

    # Just Cook County, IL
    ds = combined_datasets.datasource_regions(
        orig_data_source_cls, include=Region.from_fips("17031")
    ).make_dataset()
    assert ds.location_ids.to_list() == ["iso1:us#iso2:us-il#fips:17031"]
