import io
import pathlib
import pytest
import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields

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
    multiregion = timeseries.MultiRegionDataset.from_timeseries_and_latest(
        ts, ts.latest_values_object()
    )
    pd.testing.assert_frame_equal(
        ts.data,
        multiregion.data_with_fips.drop(columns=[CommonFields.LOCATION_ID]),
        check_like=True,
    )

    ts_again = multiregion.to_timeseries()
    pd.testing.assert_frame_equal(
        ts.data, ts_again.data.drop(columns=[CommonFields.LOCATION_ID]), check_like=True
    )


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
    multiregion = timeseries.MultiRegionDataset.from_timeseries_and_latest(
        ts, latest_values
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#fips:97111,m1,ts197111prov\n")
    )
    region_97111 = multiregion.get_one_region(Region.from_fips("97111"))
    assert region_97111.date_indexed.at["2020-04-02", "m1"] == 2
    assert region_97111.latest["c1"] == 3
    assert multiregion.get_one_region(Region.from_fips("01")).latest["c2"] == 123.4

    csv_path = tmp_path / "multiregion.csv"
    multiregion.to_csv(csv_path)
    multiregion_loaded = timeseries.MultiRegionDataset.from_csv(csv_path)
    region_97111 = multiregion_loaded.get_one_region(Region.from_fips("97111"))
    assert region_97111.date_indexed.at["2020-04-02", "m1"] == 2
    assert region_97111.latest["c1"] == 3
    assert region_97111.region.fips == "97111"
    assert multiregion_loaded.get_one_region(Region.from_fips("01")).latest["c2"] == 123.4
    assert_dataset_like(
        multiregion, multiregion_loaded, drop_na_latest=True, drop_na_timeseries=True
    )


def test_multi_region_get_one_region():
    ts = timeseries.MultiRegionDataset.from_csv(
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
    assert region_97111_ts.region.fips == "97111"

    region_97222_ts = ts.get_one_region(Region.from_fips("97222"))
    assert to_dict(["date"], region_97222_ts.data) == {
        pd.to_datetime("2020-04-01"): {
            "m2": 10,
            "county": "Foo County",
            "location_id": "iso1:us#fips:97222",
            "aggregate_level": "county",
        }
    }
    assert region_97222_ts.latest["m2"] == 11


def test_multi_region_get_counties():
    ts = timeseries.MultiRegionDataset.from_csv(
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
    assert to_dict(["location_id", "date"], counties_ts.data[["location_id", "date", "m1"]]) == {
        ("iso1:us#fips:97111", pd.to_datetime("2020-04-02")): {"m1": 2},
        ("iso1:us#fips:97111", pd.to_datetime("2020-04-03")): {"m1": 3},
    }


def test_multi_region_groupby():
    ts = timeseries.MultiRegionDataset.from_csv(
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
    bar_county_row = {
        "location_id": "iso1:us#fips:97111",
        "county": "Bar County",
        "aggregate_level": "county",
        "date": "2020-04-02",
        "m1": 2,
        "m2": pd.NA,
    }
    ts = timeseries.OneRegionTimeseriesDataset(
        Region.from_fips("97111"), pd.DataFrame([bar_county_row]), {}
    )
    assert ts.has_one_region() == True

    foo_county_row = {
        "location_id": "iso1:us#fips:97222",
        "county": "Foo County",
        "aggregate_level": "county",
        "date": "2020-04-01",
        "m1": pd.NA,
        "m2": 10,
    }
    with pytest.raises(ValueError):
        timeseries.OneRegionTimeseriesDataset(
            Region.from_fips("97222"), pd.DataFrame([bar_county_row, foo_county_row]), {},
        )

    with structlog.testing.capture_logs() as logs:
        ts = timeseries.OneRegionTimeseriesDataset(
            Region.from_fips("99999"),
            pd.DataFrame([], columns="location_id county aggregate_level date m1 m2".split()),
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
    out = timeseries.MultiRegionDataset.from_timeseries_and_latest(ts, ts.latest_values_object())
    # Use loc[...].at[...] as work-around for https://github.com/pandas-dev/pandas/issues/26989
    assert out._provenance.loc["iso1:us#fips:97111"].at["m1"] == "src11"
    assert out.get_one_region(Region.from_fips("97111")).provenance["m1"] == "src11"
    assert out._provenance.loc["iso1:us#fips:97222"].at["m2"] == "src22"
    assert out.get_one_region(Region.from_fips("97222")).provenance["m2"] == "src22"
    assert out._provenance.loc["iso1:us#fips:03"].at["m2"] == "src32"
    assert out.get_one_region(Region.from_fips("03")).provenance["m2"] == "src32"

    counties = out.get_counties(after=pd.to_datetime("2020-04-01"))
    assert "iso1:us#fips:03" not in counties._provenance.index
    assert counties._provenance.loc["iso1:us#fips:97222"].at["m1"] == "src21"
    assert counties.get_one_region(Region.from_fips("97222")).provenance["m1"] == "src21"


def _timeseries_sorted_by_location_date(
    ts: timeseries.MultiRegionDataset, drop_na: bool
) -> pd.DataFrame:
    """Returns the timeseries data, sorted by LOCATION_ID and DATE."""
    df = ts.data.sort_values([CommonFields.LOCATION_ID, CommonFields.DATE], ignore_index=True)
    if drop_na:
        df = df.dropna("columns", "all")
    return df


def _latest_sorted_by_location_date(
    ts: timeseries.MultiRegionDataset, drop_na: bool
) -> pd.DataFrame:
    """Returns the latest data, sorted by LOCATION_ID."""
    df = ts._latest_data.sort_values([CommonFields.LOCATION_ID], ignore_index=True)
    if drop_na:
        df = df.dropna("columns", "all")
    return df


def assert_dataset_like(
    ds1: timeseries.MultiRegionDataset,
    ds2: timeseries.MultiRegionDataset,
    drop_na_timeseries=False,
    drop_na_latest=False,
):
    """Asserts that two datasets contain similar date, ignoring order."""
    ts1 = _timeseries_sorted_by_location_date(ds1, drop_na=drop_na_timeseries)
    ts2 = _timeseries_sorted_by_location_date(ds2, drop_na=drop_na_timeseries)
    pd.testing.assert_frame_equal(ts1, ts2, check_like=True)
    latest1 = _latest_sorted_by_location_date(ds1, drop_na_latest)
    latest2 = _latest_sorted_by_location_date(ds2, drop_na_latest)
    pd.testing.assert_frame_equal(latest1, latest2, check_like=True, check_dtype=False)
    pd.testing.assert_series_equal(ds1._provenance, ds2._provenance)


def test_append_regions():
    ts_fips = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-03,Bar County,county,3,\n"
            "iso1:us#fips:97222,2020-04-04,Foo County,county,,11\n"
            "iso1:us#fips:97111,,Bar County,county,3,\n"
            "iso1:us#fips:97222,,Foo County,county,,11\n"
        )
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#fips:97111,m1,prov97111m1\n")
    )
    ts_cbsa = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,2\n"
            "iso1:us#cbsa:10100,2020-04-03,3\n"
            "iso1:us#cbsa:20200,2020-04-03,4\n"
            "iso1:us#cbsa:10100,,3\n"
            "iso1:us#cbsa:20200,,4\n"
        )
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#cbsa:20200,m1,prov20200m2\n")
    )
    # Check that merge is symmetric
    ts_merged_1 = ts_fips.append_regions(ts_cbsa)
    ts_merged_2 = ts_cbsa.append_regions(ts_fips)
    assert_dataset_like(ts_merged_1, ts_merged_2)

    ts_expected = timeseries.MultiRegionDataset.from_csv(
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
    ).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#fips:97111,m1,prov97111m1\n"
            "iso1:us#cbsa:20200,m1,prov20200m2\n"
        )
    )
    assert_dataset_like(ts_merged_1, ts_expected)


def test_calculate_new_cases():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:1,2020-01-01,0\n"
            "iso1:us#fips:1,2020-01-02,1\n"
            "iso1:us#fips:1,2020-01-03,1\n"
            "iso1:us#fips:2,2020-01-01,5\n"
            "iso1:us#fips:2,2020-01-02,7\n"
            "iso1:us#fips:3,2020-01-01,9\n"
            "iso1:us#fips:4,2020-01-01,\n"
            "iso1:us#fips:1,,100\n"
            "iso1:us#fips:2,,\n"
            "iso1:us#fips:3,,\n"
            "iso1:us#fips:4,,\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#fips:1,2020-01-01,0,0\n"
            "iso1:us#fips:1,2020-01-02,1,1\n"
            "iso1:us#fips:1,2020-01-03,1,0\n"
            "iso1:us#fips:2,2020-01-01,5,5\n"
            "iso1:us#fips:2,2020-01-02,7,2\n"
            "iso1:us#fips:3,2020-01-01,9,9\n"
            "iso1:us#fips:4,2020-01-01,,\n"
            "iso1:us#fips:1,,100,0.0\n"
            "iso1:us#fips:2,,,2.0\n"
            "iso1:us#fips:3,,,9.0\n"
            "iso1:us#fips:4,,,\n"
        )
    )

    timeseries_after = timeseries.add_new_cases(mrts_before)
    assert_dataset_like(mrts_expected, timeseries_after)


def test_new_cases_remove_negative():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:1,2020-01-01,100\n"
            "iso1:us#fips:1,2020-01-02,50\n"
            "iso1:us#fips:1,2020-01-03,75\n"
            "iso1:us#fips:1,,75\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#fips:1,2020-01-01,100,100\n"
            "iso1:us#fips:1,2020-01-02,50,\n"
            "iso1:us#fips:1,2020-01-03,75,25\n"
            "iso1:us#fips:1,,75,25.0\n"
        )
    )

    timeseries_after = timeseries.add_new_cases(mrts_before)
    assert_dataset_like(mrts_expected, timeseries_after)


def test_timeseries_long():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,,3\n"
            "iso1:us#cbsa:10100,,,,,3\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4,\n"
            "iso1:us#fips:97111,,Bar County,county,4,\n"
        )
    )

    expected = pd.read_csv(
        io.StringIO(
            "location_id,date,variable,value\n"
            "iso1:us#cbsa:10100,2020-04-02,m2,2\n"
            "iso1:us#cbsa:10100,2020-04-03,m2,3\n"
            "iso1:us#fips:97111,2020-04-02,m1,2\n"
            "iso1:us#fips:97111,2020-04-04,m1,4\n"
        ),
        parse_dates=[CommonFields.DATE],
        dtype={"value": float},
    )
    long = ts.timeseries_long(["m1", "m2"]).sort_values(
        [CommonFields.LOCATION_ID, PdFields.VARIABLE, CommonFields.DATE], ignore_index=True
    )
    pd.testing.assert_frame_equal(long, expected, check_like=True)


def test_join_columns():
    ts_1 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,\n"
            "iso1:us#cbsa:10100,2020-04-03,,,\n"
            "iso1:us#cbsa:10100,,,,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4\n"
            "iso1:us#fips:97111,,Bar County,county,4\n"
        )
    ).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#cbsa:10100,m1,ts110100prov\n"
            "iso1:us#fips:97111,m1,ts197111prov\n"
        )
    )
    ts_2 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,,,2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,3\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,\n"
        )
    ).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#cbsa:10100,m2,ts110100prov\n"
            "iso1:us#fips:97111,m2,ts197111prov\n"
        )
    )
    ts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,,3\n"
            "iso1:us#cbsa:10100,,,,,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4,\n"
            "iso1:us#fips:97111,,Bar County,county,4,\n"
        )
    ).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#cbsa:10100,m1,ts110100prov\n"
            "iso1:us#cbsa:10100,m2,ts110100prov\n"
            "iso1:us#fips:97111,m1,ts197111prov\n"
            "iso1:us#fips:97111,m2,ts197111prov\n"
        )
    )
    ts_joined = ts_1.join_columns(ts_2)
    assert_dataset_like(ts_joined, ts_expected, drop_na_latest=True)

    with pytest.raises(NotImplementedError):
        ts_2.join_columns(ts_1)

    with pytest.raises(ValueError):
        # Raises because the same column is in both datasets
        ts_2.join_columns(ts_2)

    # Checking geo attributes is currently disabled.
    # ts_2_variation_df = ts_2.combined_df.copy()
    # ts_2_variation_df.loc[
    #     ts_2_variation_df[CommonFields.COUNTY] == "Bar County", CommonFields.COUNTY
    # ] = "Bart County"
    # ts_2_variation = timeseries.MultiRegionDataset.from_combined_dataframe(
    #     ts_2_variation_df
    # )
    # with pytest.raises(ValueError):
    #     ts_1.join_columns(ts_2_variation)


def test_join_columns_missing_regions():
    ts_1 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,\n"
            "iso1:us#cbsa:10100,2020-04-03,,,\n"
            "iso1:us#cbsa:10100,,,,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4\n"
            "iso1:us#fips:97111,,Bar County,county,4\n"
        )
    )
    ts_2 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m2\n" "iso1:us#cbsa:10100,2020-04-02,,,2\n"
        )
    )
    ts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,,\n"
            "iso1:us#cbsa:10100,,,,,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4,\n"
            "iso1:us#fips:97111,,Bar County,county,4,\n"
        )
    )
    ts_joined = ts_1.join_columns(ts_2)
    assert_dataset_like(ts_joined, ts_expected, drop_na_latest=True)


def test_iter_one_region():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,\n"
            "iso1:us#cbsa:10100,2020-04-03,,,\n"
            "iso1:us#cbsa:10100,,,,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4\n"
            "iso1:us#fips:97111,,Bar County,county,4\n"
            # 97222 does not have a row of latest data to make sure it still works
            "iso1:us#fips:97222,2020-04-02,No Recent County,county,3\n"
            "iso1:us#fips:97222,2020-04-04,No Recent County,county,5\n"
        )
    )
    assert {region.location_id for region, _ in ts.iter_one_regions()} == {
        "iso1:us#cbsa:10100",
        "iso1:us#fips:97111",
        "iso1:us#fips:97222",
    }
    for it_region, it_one_region in ts.iter_one_regions():
        one_region = ts.get_one_region(it_region)
        assert (one_region.data.fillna("") == it_one_region.data.fillna("")).all(axis=None)
        assert one_region.latest == it_one_region.latest
        assert one_region.provenance == it_one_region.provenance
        assert one_region.region == it_region
        assert one_region.region == it_one_region.region


def test_drop_regions_without_population():
    # Only regions with location_id containing 1 have population, those with 2 don't
    ts_in = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,population,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,\n"
            "iso1:us#cbsa:10100,,,,80000,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,,2\n"
            "iso1:us#fips:97111,,Bar County,county,40000,4\n"
            "iso1:us#cbsa:20200,2020-04-02,,,,\n"
            "iso1:us#cbsa:20200,,,,,\n"
            "iso1:us#fips:97222,2020-04-02,Bar County,county,,2\n"
            "iso1:us#fips:97222,,Bar County,county,,4\n"
        )
    )
    ts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,population,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,\n"
            "iso1:us#cbsa:10100,,,,80000,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,,2\n"
            "iso1:us#fips:97111,,Bar County,county,40000,4\n"
        )
    )
    with structlog.testing.capture_logs() as logs:
        ts_out = timeseries.drop_regions_without_population(
            ts_in, ["iso1:us#fips:97222"], structlog.get_logger()
        )
    assert_dataset_like(ts_out, ts_expected)

    assert [l["event"] for l in logs] == ["Dropping unexpected regions without populaton"]
    assert [l["location_ids"] for l in logs] == [["iso1:us#cbsa:20200"]]


def test_merge_provenance():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,\n"
            "iso1:us#cbsa:10100,2020-04-03,,,\n"
            "iso1:us#cbsa:10100,,,,\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4\n"
            "iso1:us#fips:97111,,Bar County,county,4\n"
        )
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#cbsa:10100,m1,ts110100prov\n")
    )

    with pytest.raises(NotImplementedError):
        ts.add_provenance_csv(
            io.StringIO("location_id,variable,provenance\n" "iso1:us#fips:97111,m1,ts197111prov\n")
        )


def _build_one_column_multiregion_dataset(
    column, values, location_id="iso1:us#fips:97222", start_date="2020-08-25"
):

    dates = pd.date_range(start_date, periods=len(values), freq="D")
    ts_rows = []

    for i, value in enumerate(values):
        ts_rows.append({CommonFields.LOCATION_ID: location_id, "date": dates[i], column: value})
    timeseries_df = pd.DataFrame(ts_rows)

    # Find last non nan value, which simulates building the latest value entry
    for value in reversed(values):
        if not pd.isna(value) and value is not None:
            break
    latest_df = pd.DataFrame(
        [{CommonFields.LOCATION_ID: location_id, "date": pd.NaT, column: value}]
    )

    return timeseries.MultiRegionDataset.from_timeseries_df(timeseries_df).add_latest_df(latest_df)


def test_remove_outliers():
    values = [10.0] * 7 + [1000.0]
    dataset = _build_one_column_multiregion_dataset(CommonFields.NEW_CASES, values)
    dataset = timeseries.drop_new_case_outliers(dataset)

    # Expected result is the same series with the last value removed
    values = [10.0] * 7 + [None]
    expected = _build_one_column_multiregion_dataset(CommonFields.NEW_CASES, values)
    assert_dataset_like(dataset, expected)


def test_remove_outliers_threshold():
    values = [1.0] * 7 + [30.0]
    dataset = _build_one_column_multiregion_dataset(CommonFields.NEW_CASES, values)
    result = timeseries.drop_new_case_outliers(dataset, case_threshold=30)

    # Should not modify becasue not higher than threshold
    assert_dataset_like(dataset, result)

    result = timeseries.drop_new_case_outliers(dataset, case_threshold=29)

    # Expected result is the same series with the last value removed
    values = [1.0] * 7 + [None]
    expected = _build_one_column_multiregion_dataset(CommonFields.NEW_CASES, values)
    assert_dataset_like(result, expected)


def test_not_removing_short_series():
    values = [None] * 7 + [1, 1, 300]
    dataset = _build_one_column_multiregion_dataset(CommonFields.NEW_CASES, values)
    result = timeseries.drop_new_case_outliers(dataset, case_threshold=30)

    # Should not modify becasue not higher than threshold
    assert_dataset_like(dataset, result)
