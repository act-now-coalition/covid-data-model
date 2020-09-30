import io
import pathlib

import pytest
import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields

from covidactnow.datapublic.common_test_helpers import to_dict
from libs.datasets import combined_datasets

from libs.datasets import timeseries
from libs.pipeline import Region
from test.dataset_utils_test import read_csv_and_index_fips
from test.dataset_utils_test import read_csv_and_index_fips_date

# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error", "ignore::libs.pipeline.BadFipsWarning")


@pytest.mark.parametrize("include_na_at_end", [False, True])
def test_remove_padded_nans(include_na_at_end):
    rows = [
        {"date": "2020-02-01", "cases": pd.NA},
        {"date": "2020-02-02", "cases": pd.NA},
        {"date": "2020-02-03", "cases": 1},
        {"date": "2020-02-04", "cases": pd.NA},
        {"date": "2020-02-05", "cases": 2},
        {"date": "2020-02-06", "cases": 3},
    ]
    if include_na_at_end:
        rows += [{"date": "2020-02-07", "cases": pd.NA}]

    df = pd.DataFrame(rows)

    results = timeseries._remove_padded_nans(df, ["cases"])
    expected_series = pd.Series([1, pd.NA, 2, 3], name="cases")

    pd.testing.assert_series_equal(results.cases, expected_series)


def test_multi_region_to_from_timeseries():
    ts = timeseries.TimeseriesDataset(
        read_csv_and_index_fips_date(
            "fips,county,aggregate_level,date,m1,m2\n"
            "97111,Bar County,county,2020-04-02,2,\n"
            "97222,Foo County,county,2020-04-01,,10\n"
            "01,,state,2020-04-01,,20\n"
        ).reset_index()
    )
    multiregion = timeseries.MultiRegionTimeseriesDataset.from_timeseries_and_latest(
        ts, ts.latest_values_object()
    )
    pd.testing.assert_frame_equal(
        ts.data, multiregion.data.drop(columns=[CommonFields.LOCATION_ID])
    )

    ts_again = multiregion.to_timeseries()
    pd.testing.assert_frame_equal(ts.data, ts_again.data.drop(columns=[CommonFields.LOCATION_ID]))


def test_multi_region_to_from_timeseries_and_latest_values(tmp_path: pathlib.Path):
    ts = timeseries.TimeseriesDataset(
        read_csv_and_index_fips_date(
            "fips,county,aggregate_level,date,m1,m2\n"
            "97111,Bar County,county,2020-04-02,2,\n"
            "97222,Foo County,county,2020-04-01,,10\n"
            "01,,state,2020-04-01,,20\n"
        ).reset_index()
    )
    latest_values = timeseries.LatestValuesDataset(
        read_csv_and_index_fips(
            "fips,county,aggregate_level,c1,c2\n"
            "97111,Bar County,county,3,\n"
            "97222,Foo County,county,4,10.5\n"
            "01,,state,,123.4\n"
        ).reset_index()
    )
    multiregion = timeseries.MultiRegionTimeseriesDataset.from_timeseries_and_latest(
        ts, latest_values
    )
    region_97111 = multiregion.get_one_region(Region.from_fips("97111"))
    assert region_97111.date_indexed.at["2020-04-02", "m1"] == 2
    assert region_97111.latest["c1"] == 3
    assert multiregion.get_one_region(Region.from_fips("01")).latest["c2"] == 123.4

    csv_path = tmp_path / "multiregion.csv"
    multiregion.to_csv(csv_path)
    multiregion_loaded = timeseries.MultiRegionTimeseriesDataset.from_csv(csv_path)
    region_97111 = multiregion_loaded.get_one_region(Region.from_fips("97111"))
    assert region_97111.date_indexed.at["2020-04-02", "m1"] == 2
    assert region_97111.latest["c1"] == 3
    assert multiregion_loaded.get_one_region(Region.from_fips("01")).latest["c2"] == 123.4


def test_multi_region_get_one_region():
    ts = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,m2\n"
            "iso1:us#fips:97111,Bar County,county,2020-04-02,2,\n"
            "iso1:us#fips:97222,Foo County,county,2020-04-01,,10\n"
            "iso1:us#fips:97111,Bar County,county,,3,\n"
            "iso1:us#fips:97222,Foo County,county,,,11\n"
        )
    )
    region_97111_ts = ts.get_one_region(Region.from_fips("97111"))
    assert to_dict(["date"], region_97111_ts.data[["date", "m1", "m2"]]) == {
        pd.to_datetime("2020-04-02"): {"m1": 2}
    }
    assert region_97111_ts.latest["m1"] == 3

    region_97222_ts = ts.get_one_region(Region.from_fips("97222"))
    assert to_dict(["date"], region_97222_ts.data) == {
        pd.to_datetime("2020-04-01"): {
            "m2": 10,
            "county": "Foo County",
            "fips": "97222",
            "location_id": "iso1:us#fips:97222",
            "aggregate_level": "county",
        }
    }
    assert region_97222_ts.latest["m2"] == 11


def test_multi_region_get_counties():
    ts = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,m2\n"
            "iso1:us#fips:97111,Bar County,county,2020-04-02,2,\n"
            "iso1:us#fips:97111,Bar County,county,2020-04-03,3,\n"
            "iso1:us#fips:97222,Foo County,county,2020-04-01,,10\n"
            "iso1:us#fips:97,Great State,state,2020-04-01,1,2\n"
            "iso1:us#fips:97111,Bar County,county,,3,\n"
            "iso1:us#fips:97222,Foo County,county,,,10\n"
            "iso1:us#fips:97,Great State,state,,1,2\n"
        )
    )
    counties_ts = ts.get_counties(after=pd.to_datetime("2020-04-01"))
    assert to_dict(["fips", "date"], counties_ts.data[["fips", "date", "m1"]]) == {
        ("97111", pd.to_datetime("2020-04-02")): {"m1": 2},
        ("97111", pd.to_datetime("2020-04-03")): {"m1": 3},
    }


def test_multi_region_groupby():
    ts = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,m2\n"
            "iso1:us#fips:97222,Foo County,county,2020-04-01,,10\n"
            "iso1:us#fips:97222,Foo County,county,2020-04-02,,20\n"
            "iso1:us#fips:97,Great State,state,2020-04-01,1,2\n"
            "iso1:us#fips:97222,Foo County,county,,,20\n"
            "iso1:us#fips:97,Great State,state,,1,2\n"
        )
    )

    assert ts.groupby_region()["m2"].last().to_dict() == {
        "iso1:us#fips:97": 2,
        "iso1:us#fips:97222": 20,
    }


def test_one_region_dataset():
    ts = timeseries.OneRegionTimeseriesDataset(
        read_csv_and_index_fips_date(
            "fips,county,aggregate_level,date,m1,m2\n" "97111,Bar County,county,2020-04-02,2,\n"
        ).reset_index(),
        {},
    )
    assert ts.has_one_region() == True

    with pytest.raises(ValueError):
        timeseries.OneRegionTimeseriesDataset(
            read_csv_and_index_fips_date(
                "fips,county,aggregate_level,date,m1,m2\n"
                "97111,Bar County,county,2020-04-02,2,\n"
                "97222,Foo County,county,2020-04-01,,10\n"
            ).reset_index(),
            {},
        )

    with structlog.testing.capture_logs() as logs:
        ts = timeseries.OneRegionTimeseriesDataset(
            read_csv_and_index_fips_date("fips,county,aggregate_level,date,m1,m2\n").reset_index(),
            {},
        )
    assert [l["event"] for l in logs] == ["Creating OneRegionTimeseriesDataset with zero regions"]
    assert ts.empty


def test_multiregion_provenance():
    input_df = read_csv_and_index_fips_date(
        "fips,county,aggregate_level,date,m1,m2\n"
        "97111,Bar County,county,2020-04-01,1,\n"
        "97111,Bar County,county,2020-04-02,2,\n"
        "97222,Foo County,county,2020-04-01,,10\n"
        "97222,Foo County,county,2020-04-03,3,30\n"
        "03,,state,2020-04-03,4,40\n"
    )
    provenance = combined_datasets.provenance_wide_metrics_to_series(
        read_csv_and_index_fips_date(
            "fips,date,m1,m2\n"
            "97111,2020-04-01,src11,\n"
            "97111,2020-04-02,src11,\n"
            "97222,2020-04-01,,src22\n"
            "97222,2020-04-03,src21,src22\n"
            "03,2020-04-03,src31,src32\n"
        ),
        structlog.get_logger(),
    )
    ts = timeseries.TimeseriesDataset(input_df.reset_index(), provenance=provenance)
    out = timeseries.MultiRegionTimeseriesDataset.from_timeseries_and_latest(
        ts, ts.latest_values_object()
    )
    # Use loc[...].at[...] as work-around for https://github.com/pandas-dev/pandas/issues/26989
    assert out.provenance.loc["iso1:us#fips:97111"].at["m1"] == "src11"
    assert out.provenance.loc["iso1:us#fips:97222"].at["m2"] == "src22"
    assert out.provenance.loc["iso1:us#fips:03"].at["m2"] == "src32"

    counties = out.get_counties(after=pd.to_datetime("2020-04-01"))
    assert "iso1:us#fips:03" not in counties.provenance.index
    assert counties.provenance.loc["iso1:us#fips:97222"].at["m1"] == "src21"


def _combined_sorted_by_location_date(ts: timeseries.MultiRegionTimeseriesDataset) -> pd.DataFrame:
    """Returns the combined data, sorted by LOCATION_ID and DATE."""
    return ts.combined_df.sort_values(
        [CommonFields.LOCATION_ID, CommonFields.DATE], ignore_index=True
    )


def assert_combined_like(
    ts1: timeseries.MultiRegionTimeseriesDataset, ts2: timeseries.MultiRegionTimeseriesDataset
):
    """Asserts that two datasets contain similar date, ignoring order."""
    sorted1 = _combined_sorted_by_location_date(ts1)
    sorted2 = _combined_sorted_by_location_date(ts2)
    pd.testing.assert_frame_equal(sorted1, sorted2, check_like=True)


def test_merge():
    ts_fips = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-03,Bar County,county,3,\n"
            "iso1:us#fips:97222,2020-04-04,Foo County,county,,11\n"
            "iso1:us#fips:97111,,Bar County,county,3,\n"
            "iso1:us#fips:97222,,Foo County,county,,11\n"
        )
    )
    ts_cbsa = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,2\n"
            "iso1:us#cbsa:10100,2020-04-03,3\n"
            "iso1:us#cbsa:20200,2020-04-03,4\n"
            "iso1:us#cbsa:10100,,3\n"
            "iso1:us#cbsa:20200,,4\n"
        )
    )
    # Check that merge is symmetric
    ts_merged_1 = ts_fips.merge(ts_cbsa)
    ts_merged_2 = ts_cbsa.merge(ts_fips)
    assert_combined_like(ts_merged_1, ts_merged_2)

    ts_expected = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,,3\n"
            "iso1:us#cbsa:20200,2020-04-03,,,,4\n"
            "iso1:us#cbsa:10100,,,,,3\n"
            "iso1:us#cbsa:20200,,,,,4\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-03,Bar County,county,3,\n"
            "iso1:us#fips:97222,2020-04-04,Foo County,county,,11\n"
            "iso1:us#fips:97111,,Bar County,county,3,\n"
            "iso1:us#fips:97222,,Foo County,county,,11\n"
        )
    )
    assert_combined_like(ts_merged_1, ts_expected)
