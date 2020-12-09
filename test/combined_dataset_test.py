import logging
import re

import structlog

from libs.datasets import combined_datasets, CommonFields
from libs.datasets import timeseries
from libs.datasets.combined_datasets import provenance_wide_metrics_to_series
from libs.datasets.sources.covid_county_data import CovidCountyDataDataSource
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations

from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.datasets.sources.covid_tracking_source import CovidTrackingDataSource

from libs.pipeline import Region
from test.dataset_utils_test import read_csv_and_index_fips_date
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
@pytest.mark.parametrize("fips", ["06075", "48201"])
def test_combined_county_has_some_data(fips):
    region_data = combined_datasets.load_us_timeseries_dataset().get_one_region(
        Region.from_fips(fips)
    )
    assert region_data.data[CommonFields.POSITIVE_TESTS].all()
    assert region_data.data[CommonFields.NEGATIVE_TESTS].all()
    assert region_data.latest[CommonFields.DEATHS] > 1


def test_pr_aggregation():
    dataset = combined_datasets.load_us_timeseries_dataset()
    data = dataset.get_one_region(Region.from_fips("72")).latest
    assert data
    assert data["all_beds_occupancy_rate"] < 1
    assert data["icu_occupancy_rate"] < 1


def test_nyc_aggregation(nyc_region):
    dataset = combined_datasets.load_us_timeseries_dataset()
    data = dataset.get_one_region(nyc_region).latest
    # Check to make sure that beds occupancy rates are below 1,
    # signaling that it is properly combining occupancy rates.
    assert data["all_beds_occupancy_rate"] < 1
    assert data["icu_occupancy_rate"] < 1


# Check some counties picked arbitrarily: (Orange County, CA)/06059 and (Harris County, TX)/48201
@pytest.mark.parametrize("fips", ["06059", "48201"])
def test_combined_county_has_some_timeseries_data(fips):
    region = Region.from_fips(fips)
    latest = combined_datasets.load_us_timeseries_dataset().get_one_region(region)
    df = latest.data.set_index(CommonFields.DATE)
    date = "2020-09-01"  # Arbitrary date that both FIPS have data for.
    assert df.loc[date, CommonFields.CASES] > 0
    assert df.loc[date, CommonFields.DEATHS] > 0
    assert df.loc[date, CommonFields.POSITIVE_TESTS] > 0
    assert df.loc[date, CommonFields.NEGATIVE_TESTS] > 0
    assert df.loc[date, CommonFields.CURRENT_ICU] > 0


def test_get_county_name():
    assert combined_datasets.get_county_name(Region.from_fips("06059")) == "Orange County"
    assert combined_datasets.get_county_name(Region.from_fips("48201")) == "Harris County"


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_source_cls",
    [CovidTrackingDataSource, CovidCountyDataDataSource, NYTimesDataset, TexasHospitalizations,],
)
def test_unique_timeseries(data_source_cls):
    dataset = data_source_cls.local().multi_region_dataset()
    # Check for duplicate rows with the same INDEX_FIELDS. Sort by index so duplicates are next to
    # each other in the message if the assert fails.
    timeseries_data = dataset.timeseries.sort_index()
    assert timeseries_data.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
    duplicates = timeseries_data.index.duplicated(keep=False)
    assert not duplicates.any(), str(timeseries_data.loc[duplicates])


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_source_cls", [CovidTrackingDataSource, CovidCountyDataDataSource],
)
def test_expected_field_in_sources(data_source_cls):
    dataset = data_source_cls.local().multi_region_dataset()

    assert not dataset.timeseries.empty
    assert not dataset.static.empty

    states = set(dataset.static["state"])

    good_state = set()
    for state in states:
        if re.fullmatch(r"[A-Z]{2}", state):
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
        "97123,Smith County,ZZ,USA,2020-04-01,county,1,\n"
        "97123,Smith County,ZZ,USA,2020-04-02,county,,2\n"
    ).reset_index()
    ds = timeseries.MultiRegionDataset.from_fips_timeseries_df(data)
    region = ds.get_one_region(Region.from_fips("97123"))
    # Compare 2 values in region.latest
    expected = {"m1": 1, "m2": 2}
    actual = {key: region.latest[key] for key in expected.keys()}
    assert actual == expected


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
