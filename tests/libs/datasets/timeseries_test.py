import datetime
import io
import pathlib
import dataclasses
import pickle

import pytest
import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields

from covidactnow.datapublic.common_test_helpers import to_dict

from libs import github_utils
from libs.datasets import AggregationLevel
from libs.datasets import dataset_pointer
from libs.datasets import taglib

from libs.datasets import timeseries
from libs.datasets.taglib import TagType
from libs.datasets.taglib import UrlStr
from libs.pipeline import Region
from tests import test_helpers
from tests.dataset_utils_test import read_csv_and_index_fips
from tests.dataset_utils_test import read_csv_and_index_fips_date
from tests.test_helpers import TimeseriesLiteral


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error", "ignore::libs.pipeline.BadFipsWarning")


def _make_dataset_pointer(tmpdir, filename: str = "somefile.csv") -> dataset_pointer.DatasetPointer:
    # The fixture passes in a py.path, which is not the type in DatasetPointer.
    path = pathlib.Path(tmpdir) / filename

    fake_git_summary = github_utils.GitSummary(sha="abcdef", branch="main", is_dirty=True)

    return dataset_pointer.DatasetPointer(
        dataset_type=dataset_pointer.DatasetType.MULTI_REGION,
        path=path,
        data_git_info=fake_git_summary,
        model_git_info=fake_git_summary,
        updated_at=datetime.datetime.utcnow(),
    )


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


def test_multi_region_to_from_timeseries_and_latest_values(tmp_path: pathlib.Path):
    ts_df = read_csv_and_index_fips_date(
        "fips,county,aggregate_level,date,m1,m2\n"
        "97111,Bar County,county,2020-04-02,2,\n"
        "97222,Foo County,county,2020-04-01,,10\n"
        "01,,state,2020-04-01,,20\n"
    ).reset_index()
    latest_values_df = read_csv_and_index_fips(
        "fips,county,aggregate_level,c1,c2\n"
        "97111,Bar County,county,3,\n"
        "97222,Foo County,county,4,10.5\n"
        "01,,state,,123.4\n"
    ).reset_index()
    multiregion = (
        timeseries.MultiRegionDataset.from_fips_timeseries_df(ts_df)
        .add_fips_static_df(latest_values_df)
        .add_provenance_csv(
            io.StringIO("location_id,variable,provenance\n" "iso1:us#fips:97111,m1,ts197111prov\n")
        )
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
    test_helpers.assert_dataset_like(
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
        pd.to_datetime("2020-04-01"): {"m2": 10, "location_id": "iso1:us#fips:97222",}
    }
    assert region_97222_ts.latest["m2"] == 11


def test_multi_region_get_counties_and_places():
    ds_in = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,m2\n"
            "iso1:us#fips:97111,Bar County,county,2020-04-02,2,\n"
            "iso1:us#fips:97111,Bar County,county,2020-04-03,3,\n"
            "iso1:us#fips:97222,Foo County,county,2020-04-01,,10\n"
            "iso1:us#fips:9711122,,place,2020-04-02,5,60\n"
            "iso1:us#fips:97,Great State,state,2020-04-01,1,2\n"
            "iso1:us#fips:97111,Bar County,county,,3,\n"
            "iso1:us#fips:9711122,,place,,3,\n"
            "iso1:us#fips:97222,Foo County,county,,,10\n"
            "iso1:us#fips:97,Great State,state,,1,2\n"
        )
    )
    ds_out = ds_in.get_counties_and_places(
        after=pd.to_datetime("2020-04-01")
    ).timeseries.reset_index()
    assert to_dict(["location_id", "date"], ds_out[["location_id", "date", "m1"]]) == {
        ("iso1:us#fips:97111", pd.to_datetime("2020-04-02")): {"m1": 2},
        ("iso1:us#fips:97111", pd.to_datetime("2020-04-03")): {"m1": 3},
        ("iso1:us#fips:9711122", pd.to_datetime("2020-04-02")): {"m1": 5},
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
    m1 = FieldName("m1")
    m2 = FieldName("m2")

    region_97111 = Region.from_fips("97111")
    region_97222 = Region.from_fips("97222")
    region_03 = Region.from_fips("03")
    ds = test_helpers.build_dataset(
        {
            region_97111: {m1: TimeseriesLiteral([1, 2, None], provenance="src11"),},
            region_97222: {
                m1: TimeseriesLiteral([None, None, 3], provenance="src21"),
                m2: TimeseriesLiteral([10, None, 30], provenance="src22"),
            },
            region_03: {
                m1: TimeseriesLiteral([None, None, 4], provenance="src31"),
                m2: TimeseriesLiteral([None, None, 40], provenance="src32"),
            },
        },
        static_by_region_then_field_name={
            region_97111: {CommonFields.AGGREGATE_LEVEL: AggregationLevel.COUNTY.value},
            region_97222: {CommonFields.AGGREGATE_LEVEL: AggregationLevel.COUNTY.value},
            region_03: {CommonFields.AGGREGATE_LEVEL: AggregationLevel.STATE.value},
        },
    )

    # Use loc[...].at[...] as work-around for https://github.com/pandas-dev/pandas/issues/26989
    assert ds.provenance.loc[region_97111.location_id].at["m1"] == "src11"
    assert ds.get_one_region(region_97111).provenance["m1"] == ["src11"]
    assert ds.provenance.loc[region_97222.location_id].at["m2"] == "src22"
    assert ds.get_one_region(region_97222).provenance["m2"] == ["src22"]
    assert ds.provenance.loc[region_03.location_id].at["m2"] == "src32"
    assert ds.get_one_region(region_03).provenance["m2"] == ["src32"]

    counties = ds.get_counties_and_places(after=pd.to_datetime("2020-04-01"))
    assert region_03.location_id not in counties.provenance.index
    assert counties.provenance.loc[region_97222.location_id].at["m1"] == "src21"
    assert counties.get_one_region(region_97222).provenance["m1"] == ["src21"]


def test_one_region_multiple_provenance():
    tag1 = test_helpers.make_tag(date="2020-04-01")
    tag2 = test_helpers.make_tag(date="2020-04-02")
    one_region = test_helpers.build_one_region_dataset(
        {
            CommonFields.ICU_BEDS: TimeseriesLiteral(
                [0, 2, 4], annotation=[tag1, tag2], provenance=["prov1", "prov2"],
            ),
            CommonFields.CASES: [100, 200, 300],
        }
    )

    assert set(one_region.annotations(CommonFields.ICU_BEDS)) == {tag1, tag2}
    assert sorted(one_region.provenance[CommonFields.ICU_BEDS]) == ["prov1", "prov2"]


def test_add_aggregate_level():
    ts_df = read_csv_and_index_fips_date("fips,date,m1,m2\n" "36061,2020-04-02,2,\n").reset_index()
    multiregion = timeseries.MultiRegionDataset.from_fips_timeseries_df(ts_df)
    assert multiregion.static.aggregate_level.to_list() == ["county"]


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
    test_helpers.assert_dataset_like(ts_merged_1, ts_merged_2)

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
    test_helpers.assert_dataset_like(ts_merged_1, ts_expected)


def test_append_regions_duplicate_region_raises():
    ts1 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
        )
    )
    ts2 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#fips:97111,2020-04-03,Bar County,county,2,\n"
        )
    )
    with pytest.raises(ValueError):
        ts1.append_regions(ts2)


def test_timeseries_long():
    region_cbsa = Region.from_cbsa_code("10100")
    region_county = Region.from_fips("97111")
    ds = test_helpers.build_dataset(
        {
            region_county: {FieldName("m1"): [2, None, 4]},
            region_cbsa: {FieldName("m2"): [2, 3, None]},
        },
        start_date="2020-04-02",
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
    long_series = ds._timeseries_long()
    assert long_series.index.names == [
        CommonFields.LOCATION_ID,
        CommonFields.DATE,
        PdFields.VARIABLE,
    ]
    assert long_series.name == PdFields.VALUE
    long_df = long_series.reset_index()
    pd.testing.assert_frame_equal(long_df, expected, check_like=True)


def test_timeseries_wide_dates():
    ds = timeseries.MultiRegionDataset.from_csv(
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

    ds_wide = ds.timeseries_not_bucketed_wide_dates
    assert ds_wide.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
    assert ds_wide.columns.names == [CommonFields.DATE]

    expected = (
        pd.read_csv(
            io.StringIO(
                "location_id,variable,2020-04-02,2020-04-03,2020-04-04\n"
                "iso1:us#cbsa:10100,m2,2,3,\n"
                "iso1:us#fips:97111,m1,2,,4\n"
            ),
        )
        .set_index(ds_wide.index.names)
        .rename_axis(columns="date")
        .astype(float)
    )
    expected.columns = pd.to_datetime(expected.columns)

    pd.testing.assert_frame_equal(ds_wide, expected)

    # Recreate the dataset using `from_timeseries_wide_dates_df`.
    ds_recreated = timeseries.MultiRegionDataset.from_timeseries_wide_dates_df(
        ds_wide
    ).add_static_values(ds.static.reset_index())
    test_helpers.assert_dataset_like(ds, ds_recreated)


def test_timeseries_wide_dates_empty():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,,,,,3\n"
            "iso1:us#fips:97111,,Bar County,county,4,\n"
        )
    )

    timeseries_wide = ts.timeseries_not_bucketed_wide_dates
    assert timeseries_wide.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
    assert timeseries_wide.columns.names == [CommonFields.DATE]
    assert timeseries_wide.empty


def test_write_read_wide_dates_csv_compare_literal(tmpdir):
    pointer = _make_dataset_pointer(tmpdir)

    region_as = Region.from_state("AS")
    region_sf = Region.from_fips("06075")
    metrics_as = {
        CommonFields.ICU_BEDS: TimeseriesLiteral([0, 2, 4], provenance="pt_src1"),
        CommonFields.CASES: [100, 200, 300],
    }
    metrics_sf = {
        CommonFields.DEATHS: TimeseriesLiteral([1, 2, None], provenance="pt_src2"),
        CommonFields.CASES: [None, 210, 310],
    }
    dataset_in = test_helpers.build_dataset({region_as: metrics_as, region_sf: metrics_sf})

    dataset_in.write_to_dataset_pointer(pointer)

    # Compare written file with a string literal so a test fails if something changes in how the
    # file is written. The literal contains spaces to align the columns in the source.
    assert pointer.path_wide_dates().read_text() == (
        "                  location_id,variable,provenance,2020-04-03,2020-04-02,2020-04-01\n"
        "           iso1:us#iso2:us-as,   cases,          ,       300,       200,       100\n"
        "           iso1:us#iso2:us-as,icu_beds,   pt_src1,         4,         2,         0\n"
        "iso1:us#iso2:us-ca#fips:06075,   cases,          ,       310,       210\n"
        "iso1:us#iso2:us-ca#fips:06075,  deaths,   pt_src2,          ,         2,         1\n"
    ).replace(" ", "")

    dataset_read = timeseries.MultiRegionDataset.read_from_pointer(pointer)

    test_helpers.assert_dataset_like(dataset_read, dataset_in)


def test_write_read_wide_dates_csv_with_annotation(tmpdir):
    pointer = _make_dataset_pointer(tmpdir)

    region = Region.from_state("AS")
    metrics = {
        CommonFields.ICU_BEDS: TimeseriesLiteral(
            [0, 2, 4],
            annotation=[
                test_helpers.make_tag(date="2020-04-01"),
                test_helpers.make_tag(TagType.ZSCORE_OUTLIER, date="2020-04-02"),
            ],
        ),
        CommonFields.CASES: [100, 200, 300],
    }
    dataset_in = test_helpers.build_dataset({region: metrics})

    dataset_in.write_to_dataset_pointer(pointer)
    dataset_read = timeseries.MultiRegionDataset.read_from_pointer(pointer)

    test_helpers.assert_dataset_like(dataset_read, dataset_in)


def test_write_read_dataset_pointer_with_provenance_list(tmpdir):
    pointer = _make_dataset_pointer(tmpdir)

    dataset_in = test_helpers.build_default_region_dataset(
        {
            CommonFields.ICU_BEDS: TimeseriesLiteral(
                [0, 2, 4],
                annotation=[
                    test_helpers.make_tag(date="2020-04-01"),
                    test_helpers.make_tag(date="2020-04-02"),
                ],
                provenance=["prov1", "prov2"],
            ),
            CommonFields.CASES: [100, 200, 300],
        }
    )

    dataset_in.write_to_dataset_pointer(pointer)
    dataset_read = timeseries.MultiRegionDataset.read_from_pointer(pointer)

    test_helpers.assert_dataset_like(dataset_read, dataset_in)


def test_timeseries_drop_stale_timeseries_entire_region():
    ds_in = timeseries.MultiRegionDataset.from_csv(
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

    ds_out = ds_in.drop_stale_timeseries(pd.to_datetime("2020-04-04"))

    ds_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,,,,,3\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4,\n"
            "iso1:us#fips:97111,,Bar County,county,4,\n"
        )
    )
    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_timeseries_drop_stale_timeseries_one_metric():
    csv_in = (
        "location_id,date,county,aggregate_level,m1,m2\n"
        "iso1:us#cbsa:10100,2020-04-02,,,11,2\n"
        "iso1:us#cbsa:10100,2020-04-03,,,,3\n"
        "iso1:us#cbsa:10100,,,,,3\n"
        "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
        "iso1:us#fips:97111,2020-04-04,Bar County,county,4,\n"
        "iso1:us#fips:97111,,Bar County,county,4,\n"
    )
    ds_in = timeseries.MultiRegionDataset.from_csv(io.StringIO(csv_in)).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#cbsa:10100,m1,m1-10100prov\n"
            "iso1:us#cbsa:10100,m2,m2-10100prov\n"
            "iso1:us#fips:97111,m1,m1-97111prov\n"
        )
    )

    ds_out = ds_in.drop_stale_timeseries(pd.to_datetime("2020-04-03"))

    # The only timeseries that is stale with cutoff of 4/3 is the CBSA m1. The expected
    # dataset is the same as the input with "11" removed from the timeseries and
    # corresponding provenance removed.
    ds_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(csv_in.replace(",11,", ",,"))
    ).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#cbsa:10100,m2,m2-10100prov\n"
            "iso1:us#fips:97111,m1,m1-97111prov\n"
        )
    )
    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_timeseries_drop_stale_timeseries_with_tag():
    region = Region.from_state("TX")
    values_recent = [100, 200, 300, 400]
    values_stale = [100, 200, None, None]
    ts_recent = TimeseriesLiteral(values_recent, annotation=[test_helpers.make_tag()])
    ts_stale = TimeseriesLiteral(values_stale, annotation=[test_helpers.make_tag()])

    dataset_in = test_helpers.build_dataset(
        {region: {CommonFields.CASES: ts_recent, CommonFields.DEATHS: ts_stale}}
    )

    dataset_out = dataset_in.drop_stale_timeseries(pd.to_datetime("2020-04-03"))

    assert len(dataset_out.tag) == 1
    # drop_stale_timeseries preserves the empty DEATHS column so add it to dataset_expected
    dataset_expected = test_helpers.build_dataset(
        {region: {CommonFields.CASES: ts_recent}}, timeseries_columns=[CommonFields.DEATHS]
    )
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_append_region_and_get_regions_subset_with_tag():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    values = [100, 200, 300, 400]
    ts_with_tag = TimeseriesLiteral(values, annotation=[test_helpers.make_tag()])

    dataset_tx = test_helpers.build_dataset({region_tx: {CommonFields.CASES: ts_with_tag}})
    dataset_sf = test_helpers.build_dataset({region_sf: {CommonFields.CASES: ts_with_tag}})

    dataset_appended = dataset_tx.append_regions(dataset_sf)

    assert len(dataset_appended.tag) == 2
    dataset_tx_and_sf = test_helpers.build_dataset(
        {region_tx: {CommonFields.CASES: ts_with_tag}, region_sf: {CommonFields.CASES: ts_with_tag}}
    )
    test_helpers.assert_dataset_like(dataset_appended, dataset_tx_and_sf)

    dataset_out = dataset_tx_and_sf.get_regions_subset([region_tx])
    assert len(dataset_out.tag) == 1
    test_helpers.assert_dataset_like(dataset_out, dataset_tx)


def test_one_region_annotations():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    values = [100, 200, 300, 400]
    tag1 = test_helpers.make_tag(date="2020-04-01")
    tag2a = test_helpers.make_tag(date="2020-04-02")
    tag2b = test_helpers.make_tag(date="2020-04-03")

    dataset_tx_and_sf = test_helpers.build_dataset(
        {
            region_tx: {CommonFields.CASES: (TimeseriesLiteral(values, annotation=[tag1]))},
            region_sf: {CommonFields.CASES: (TimeseriesLiteral(values, annotation=[tag2a, tag2b]))},
        }
    )

    # get_one_region and iter_one_regions use separate code to split up the tags. Test both of them.
    one_region_tx = dataset_tx_and_sf.get_one_region(region_tx)
    assert one_region_tx.annotations(CommonFields.CASES) == [tag1]
    one_region_sf = dataset_tx_and_sf.get_one_region(region_sf)
    assert one_region_sf.annotations(CommonFields.CASES) == [
        tag2a,
        tag2b,
    ]
    assert set(one_region_sf.sources(CommonFields.CASES)) == set()

    assert {
        region: one_region_dataset.annotations(CommonFields.CASES)
        for region, one_region_dataset in dataset_tx_and_sf.iter_one_regions()
    } == {region_sf: [tag2a, tag2b], region_tx: [tag1],}


def test_one_region_empty_annotations():
    one_region = test_helpers.build_one_region_dataset({CommonFields.CASES: [100, 200, 300]})

    assert one_region.annotations(CommonFields.CASES) == []
    assert one_region.source_url == {}
    assert one_region.provenance == {}
    assert set(one_region.sources(CommonFields.ICU_BEDS)) == set()
    assert set(one_region.sources(CommonFields.CASES)) == set()


def test_one_region_tag_objects_series():
    values = [100, 200]
    tag1 = test_helpers.make_tag(TagType.ZSCORE_OUTLIER, date="2020-04-01")
    tag2a = test_helpers.make_tag(date="2020-04-02")
    tag2b = test_helpers.make_tag(date="2020-04-03")

    one_region = test_helpers.build_one_region_dataset(
        {
            CommonFields.CASES: TimeseriesLiteral(values, annotation=[tag1]),
            CommonFields.ICU_BEDS: TimeseriesLiteral(values, provenance="prov1"),
            CommonFields.DEATHS: TimeseriesLiteral(values, annotation=[tag2a, tag2b]),
        }
    )

    assert isinstance(one_region.tag_objects_series, pd.Series)
    assert one_region.tag.index.equals(one_region.tag_objects_series.index)
    assert set(one_region.tag_objects_series.reset_index().itertuples(index=False)) == {
        (CommonFields.CASES, tag1.tag_type, tag1),
        (CommonFields.ICU_BEDS, "provenance", taglib.ProvenanceTag(source="prov1")),
        (CommonFields.DEATHS, tag2a.tag_type, tag2a),
        (CommonFields.DEATHS, tag2b.tag_type, tag2b),
    }


def test_one_region_tag_objects_series_empty():
    one_region = test_helpers.build_one_region_dataset({CommonFields.CASES: [1, 2, 3]})
    assert one_region.tag.empty
    assert isinstance(one_region.tag_objects_series, pd.Series)
    assert one_region.tag_objects_series.empty


def test_timeseries_latest_values():
    dataset = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,10,3\n"
            "iso1:us#cbsa:10100,2020-04-04,,,,1\n"
            "iso1:us#cbsa:10100,,,,,4\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4,\n"
            "iso1:us#fips:97111,,Bar County,county,5,\n"
        )
    )

    # Check bulk access via _timeseries_latest_values
    expected = pd.read_csv(
        io.StringIO("location_id,m1,m2\n" "iso1:us#cbsa:10100,10,1\n" "iso1:us#fips:97111,4,\n")
    )
    latest_from_timeseries = dataset._timeseries_latest_values().reset_index()
    pd.testing.assert_frame_equal(
        latest_from_timeseries, expected, check_like=True, check_dtype=False
    )

    # Check access to timeseries latests values via get_one_region
    region_10100 = dataset.get_one_region(Region.from_cbsa_code("10100"))
    assert region_10100.latest == {
        "aggregate_level": None,
        "county": None,
        "m1": 10,  # Derived from timeseries
        "m2": 4,  # Explicitly in recent values
    }
    region_97111 = dataset.get_one_region(Region.from_fips("97111"))
    assert region_97111.latest == {
        "aggregate_level": "county",
        "county": "Bar County",
        "m1": 5,
        "m2": None,
    }


def test_timeseries_latest_values_copied_to_static():
    dataset = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,t1,s1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,,2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,10,3\n"
            "iso1:us#cbsa:10100,2020-04-04,,,,1\n"
            "iso1:us#cbsa:10100,,,,,4\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2,\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4,\n"
            "iso1:us#fips:97111,,Bar County,county,,\n"
        )
    )

    # Check access to latest values as copied to static
    t1 = FieldName("t1")
    s1 = FieldName("s1")
    dataset_t1_latest_in_static = dataset.latest_in_static(t1)
    assert dataset_t1_latest_in_static.static.loc["iso1:us#cbsa:10100", t1] == 10
    assert dataset_t1_latest_in_static.static.loc["iso1:us#fips:97111", t1] == 4

    # Trying to copy the latest values of s1 fails because s1 already has a real value in static.
    # See also longer comment where the ValueError is raised.
    with pytest.raises(ValueError):
        dataset.latest_in_static(s1)


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
    test_helpers.assert_dataset_like(ts_joined, ts_expected, drop_na_latest=True)

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
    test_helpers.assert_dataset_like(ts_joined, ts_expected, drop_na_latest=True)


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
    test_helpers.assert_dataset_like(ts_out, ts_expected)

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


def test_append_tags():
    region_sf = Region.from_fips("06075")
    cases_values = [100, 200, 300, 400]
    metrics_sf = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([1, 2, 3, 4], provenance="pt_src2"),
        CommonFields.CASES: cases_values,
    }
    dataset_in = test_helpers.build_dataset({region_sf: metrics_sf})
    tag_sf_cases = test_helpers.make_tag(TagType.CUMULATIVE_TAIL_TRUNCATED, date="2020-04-02")
    tag_df = test_helpers.make_tag_df(region_sf, CommonFields.CASES, [tag_sf_cases])
    dataset_out = dataset_in.append_tag_df(tag_df)
    metrics_sf[CommonFields.CASES] = TimeseriesLiteral(cases_values, annotation=[tag_sf_cases])
    dataset_expected = test_helpers.build_dataset({region_sf: metrics_sf})

    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_add_provenance_all_with_tags():
    """Checks that add_provenance_all (and add_provenance_series that it calls) preserves tags."""
    region = Region.from_state("TX")
    cases_values = [100, 200, 300, 400]
    timeseries = TimeseriesLiteral(cases_values, annotation=[(test_helpers.make_tag())])
    dataset_in = test_helpers.build_dataset({region: {CommonFields.CASES: timeseries}})

    dataset_out = dataset_in.add_provenance_all("prov_prov")

    timeseries = dataclasses.replace(timeseries, provenance=["prov_prov"])
    dataset_expected = test_helpers.build_dataset({region: {CommonFields.CASES: timeseries}})

    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_join_columns_with_tags():
    """Checks that join_columns preserves tags."""
    region = Region.from_state("TX")
    cases_values = [100, 200, 300, 400]
    ts_lit = TimeseriesLiteral(cases_values, annotation=[test_helpers.make_tag()])
    dataset_cases = test_helpers.build_dataset({region: {CommonFields.CASES: ts_lit}})
    dataset_deaths = test_helpers.build_dataset({region: {CommonFields.DEATHS: ts_lit}})

    dataset_out = dataset_cases.join_columns(dataset_deaths)

    assert len(dataset_out.tag) == 2
    # The following checks that the tags in `ts_lit` have been preserved.
    dataset_expected = test_helpers.build_dataset(
        {region: {CommonFields.CASES: ts_lit, CommonFields.DEATHS: ts_lit}}
    )

    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_drop_column_with_tags():
    """Checks that join_columns preserves tags."""
    region = Region.from_state("TX")
    cases_values = [100, 200, 300, 400]
    ts_lit = TimeseriesLiteral(cases_values, annotation=[test_helpers.make_tag()])

    dataset_in = test_helpers.build_dataset(
        {region: {CommonFields.CASES: ts_lit, CommonFields.DEATHS: ts_lit}}
    )

    dataset_out = dataset_in.drop_column_if_present(CommonFields.DEATHS)

    assert len(dataset_out.tag) == 1
    dataset_expected = test_helpers.build_dataset({region: {CommonFields.CASES: ts_lit}})
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_timeseries_empty_timeseries_and_static():
    # Check that empty dataset creates a MultiRegionDataset
    # and that get_one_region raises expected exception.
    dataset = timeseries.MultiRegionDataset.new_without_timeseries()
    with pytest.raises(timeseries.RegionLatestNotFound):
        dataset.get_one_region(Region.from_fips("01001"))


def test_timeseries_empty():
    # Check that empty geodata_timeseries_df creates a MultiRegionDataset
    # and that get_one_region raises expected exception.
    dataset = timeseries.MultiRegionDataset.from_geodata_timeseries_df(
        pd.DataFrame([], columns=[CommonFields.LOCATION_ID, CommonFields.DATE])
    )
    with pytest.raises(timeseries.RegionLatestNotFound):
        dataset.get_one_region(Region.from_fips("01001"))


def test_timeseries_empty_static_not_empty():
    # Check that empty timeseries does not prevent static data working as expected.
    dataset = timeseries.MultiRegionDataset.from_geodata_timeseries_df(
        pd.DataFrame([], columns=[CommonFields.LOCATION_ID, CommonFields.DATE])
    ).add_static_values(pd.DataFrame([{"location_id": "iso1:us#fips:97111", "m1": 1234}]))
    assert dataset.get_one_region(Region.from_fips("97111")).latest["m1"] == 1234


def test_combined_timeseries():
    ds1 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,2.2\n"
            "iso1:us#cbsa:10100,2020-04-03,,,3.3\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4\n"
        )
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#cbsa:10100,m1,ds110100prov\n")
    )
    ds2 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,,,333\n"
            "iso1:us#cbsa:10100,2020-04-03,,,333\n"
            "iso1:us#fips:97222,2020-04-03,Foo County,county,30\n"
            "iso1:us#fips:97222,2020-04-04,Foo County,county,40\n"
        )
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#cbsa:10100,m1,ds110100prov\n")
    )
    combined = timeseries.combined_datasets({FieldName("m1"): [ds1, ds2]}, {})
    expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,m1\n"
            "iso1:us#cbsa:10100,2020-04-02,2.2\n"
            "iso1:us#cbsa:10100,2020-04-03,3.3\n"
            "iso1:us#fips:97111,2020-04-02,2\n"
            "iso1:us#fips:97111,2020-04-04,4\n"
            "iso1:us#fips:97222,2020-04-03,30\n"
            "iso1:us#fips:97222,2020-04-04,40\n"
        )
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#cbsa:10100,m1,ds110100prov\n")
    )

    test_helpers.assert_dataset_like(expected, combined)


def test_combined_annotation():
    ts1a = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
            test_helpers.make_tag(date="2020-04-02"),
        ],
    )
    ts1b = [100, 200, 300]
    ds1 = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts1a, CommonFields.CASES: ts1b}
    )
    ts2a = TimeseriesLiteral([1, 3, 5], annotation=[test_helpers.make_tag(date="2020-04-01")],)
    ts2b = [150, 250, 350]
    ds2 = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts2a, CommonFields.CASES: ts2b}
    )
    combined = timeseries.combined_datasets(
        {CommonFields.ICU_BEDS: [ds1, ds2], CommonFields.CASES: [ds2, ds1]}, {}
    )

    expected = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts1a, CommonFields.CASES: ts2b}
    )

    test_helpers.assert_dataset_like(combined, expected)


def test_combined_missing_field():
    ts1 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,2\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,4\n"
        )
    )
    ts2 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m2\n"
            "iso1:us#fips:97111,2020-04-02,Bar County,county,111\n"
            "iso1:us#fips:97111,2020-04-04,Bar County,county,111\n"
        )
    )
    # m1 is output, m2 is dropped.
    field_source_map = {FieldName("m1"): [ts1, ts2]}

    # Check that combining finishes and produces the expected result.
    combined_1 = timeseries.combined_datasets(field_source_map, {})
    expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,m1\n"
            "iso1:us#fips:97111,2020-04-02,2\n"
            "iso1:us#fips:97111,2020-04-04,4\n"
        )
    )
    test_helpers.assert_dataset_like(expected, combined_1)

    # Because there is only one source for the output timeseries reversing the source list
    # produces the same output.
    combined_2 = timeseries.combined_datasets(
        {name: list(reversed(source_list)) for name, source_list in field_source_map.items()}, {}
    )
    test_helpers.assert_dataset_like(expected, combined_2)


def test_combined_static():
    ds1 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,s1\n"
            "iso1:us#cbsa:10100,,,,\n"
            "iso1:us#fips:97222,,Foo County,county,22\n"
        )
    )
    ds2 = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,s1\n"
            "iso1:us#cbsa:10100,,,,111\n"
            "iso1:us#fips:97222,,Foo County,county,222\n"
        )
    )
    combined = timeseries.combined_datasets({}, {FieldName("s1"): [ds1, ds2]})
    expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO("location_id,date,s1\n" "iso1:us#cbsa:10100,,111\n" "iso1:us#fips:97222,,22\n")
    )

    test_helpers.assert_dataset_like(expected, combined, drop_na_timeseries=True)


def test_timeseries_rows():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,m2,population\n"
            "iso1:us#iso2:us-tx,Texas,state,2020-04-01,4,2,\n"
            "iso1:us#iso2:us-tx,Texas,state,2020-04-02,4,4,\n"
            "iso1:us#iso2:us-tx,Texas,state,,,,2500\n"
            "iso1:us#iso2:us-az,Arizona,state,2020-04-01,8,20,\n"
            "iso1:us#iso2:us-az,Arizona,state,2020-04-02,12,40,\n"
            "iso1:us#iso2:us-az,Arizona,state,,,,7500\n"
        )
    )

    rows = ts.timeseries_rows()
    expected = pd.read_csv(
        io.StringIO(
            "       location_id,variable,2020-04-02,2020-04-01\n"
            "iso1:us#iso2:us-az,      m1,        12,         8\n"
            "iso1:us#iso2:us-az,      m2,        40,        20\n"
            "iso1:us#iso2:us-tx,      m1,         4,         4\n"
            "iso1:us#iso2:us-tx,      m2,         4,         2\n".replace(" ", "")
        )
    ).set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE])
    pd.testing.assert_frame_equal(rows, expected, check_dtype=False, check_exact=False)


def test_multi_region_dataset_get_subset():
    ds = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,aggregate_level,state,fips,date,m1,m2,population\n"
            "iso1:us,country,,,2020-04-01,100,200,\n"
            "iso1:us,country,,,,,,10000\n"
            "iso1:us#iso2:us-tx,state,TX,,2020-04-01,4,2,\n"
            "iso1:us#iso2:us-tx,state,TX,,,,,5000\n"
            "iso1:us#fips:97222,county,,97222,2020-04-01,1,2,\n"
            "iso1:us#fips:97222,county,,97222,,,,1000\n"
            "iso1:us#cbsa:10100,cbsa,,,2020-04-01,1,2,20000\n"
        )
    )

    subset = ds.get_subset(aggregation_level=AggregationLevel.COUNTRY)
    assert subset.static.at["iso1:us", CommonFields.POPULATION] == 10000

    subset = ds.get_subset(fips="97222")
    assert subset.timeseries.at[("iso1:us#fips:97222", "2020-04-01"), "m2"] == 2

    subset = ds.get_subset(state="TX")
    assert subset.static.at["iso1:us#iso2:us-tx", CommonFields.POPULATION] == 5000

    subset = ds.get_subset(states=["TX"])
    assert subset.static.at["iso1:us#iso2:us-tx", CommonFields.POPULATION] == 5000

    subset = ds.get_subset(location_id_matches=r"\A(iso1\:us|iso1\:us\#cbsa.+)\Z")
    assert {r.location_id for r, _ in subset.iter_one_regions()} == {
        "iso1:us",
        "iso1:us#cbsa:10100",
    }


def test_dataset_regions_property(nyc_region):
    az_region = Region.from_state("AZ")
    dataset = test_helpers.build_dataset(
        {nyc_region: {CommonFields.CASES: [100]}, az_region: {CommonFields.CASES: [100]}}
    )

    assert dataset.timeseries_regions == set([az_region, nyc_region])


def test_provenance_map():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    tag1 = test_helpers.make_tag(date="2020-04-01")
    tag2 = test_helpers.make_tag(date="2020-04-02")
    dataset = test_helpers.build_dataset(
        {
            region_tx: {
                CommonFields.ICU_BEDS: TimeseriesLiteral(
                    [2, 4], annotation=[tag1, tag2], provenance=["prov1", "prov2"],
                ),
                CommonFields.CASES: TimeseriesLiteral([200, 300], provenance="prov1"),
            },
            region_sf: {
                CommonFields.ICU_BEDS: TimeseriesLiteral([2, 4], provenance="prov2",),
                CommonFields.CASES: TimeseriesLiteral([200, 300], provenance="prov3"),
                CommonFields.DEATHS: TimeseriesLiteral([1, 2], provenance="prov1"),
            },
        }
    )

    assert dataset.provenance_map() == {
        CommonFields.ICU_BEDS: {"prov1", "prov2"},
        CommonFields.CASES: {"prov1", "prov3"},
        CommonFields.DEATHS: {"prov1"},
    }


def test_provenance_map_empty():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    tag1 = test_helpers.make_tag(date="2020-04-01")
    tag2 = test_helpers.make_tag(date="2020-04-02")
    dataset = test_helpers.build_dataset(
        {
            region_tx: {
                CommonFields.ICU_BEDS: TimeseriesLiteral([2, 4], annotation=[tag1, tag2]),
                CommonFields.CASES: [200, 300],
            },
            region_sf: {CommonFields.ICU_BEDS: [2, 4],},
        }
    )

    assert dataset.provenance_map() == {}


def test_write_read_dataset_pointer_with_source_url(tmpdir):
    pointer = _make_dataset_pointer(tmpdir)
    url_str1 = UrlStr("http://foo.com/1")
    url_str2 = UrlStr("http://foo.com/2")
    url_str3 = UrlStr("http://foo.com/3")

    ts1a = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
            test_helpers.make_tag(date="2020-04-02"),
        ],
        source_url=url_str1,
    )
    ts1b = TimeseriesLiteral([100, 200, 300], source_url=[url_str2, url_str3])
    dataset_in = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts1a, CommonFields.CASES: ts1b}
    )

    dataset_in.write_to_dataset_pointer(pointer)

    dataset_read = timeseries.MultiRegionDataset.read_from_pointer(pointer)

    test_helpers.assert_dataset_like(dataset_read, dataset_in)
    source_url_read = dataset_read.get_one_region(test_helpers.DEFAULT_REGION).source_url
    assert source_url_read[CommonFields.ICU_BEDS] == [url_str1]
    # Copy to a set because the order of the URLs in the source_url may change.
    assert set(source_url_read[CommonFields.CASES]) == {url_str2, url_str3}


def test_pickle():
    ts = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
            test_helpers.make_tag(date="2020-04-02"),
        ],
        source_url=UrlStr("http://public.com"),
    )
    ds_in = test_helpers.build_default_region_dataset({CommonFields.CASES: ts})

    ds_out = pickle.loads(pickle.dumps(ds_in))

    test_helpers.assert_dataset_like(ds_in, ds_out)


def test_make_source_tags():
    url_str = UrlStr("http://foo.com/1")

    ts_prov_only = TimeseriesLiteral(
        [0, 2, 4], annotation=[test_helpers.make_tag(date="2020-04-01"),], provenance="prov_only",
    )
    ts_with_url = TimeseriesLiteral([3, 5, 7], provenance="prov_with_url", source_url=url_str)
    dataset_in = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only, CommonFields.CASES: ts_with_url}
    )

    dataset_out = timeseries.make_source_tags(dataset_in)

    source_tag_prov_only = taglib.Source("prov_only")
    ts_prov_only_expected = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[test_helpers.make_tag(date="2020-04-01"),],
        source=source_tag_prov_only,
    )
    source_tag_prov_with_url = taglib.Source("prov_with_url", url=url_str)
    ts_with_url_expected = TimeseriesLiteral([3, 5, 7], source=source_tag_prov_with_url,)
    dataset_expected = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only_expected, CommonFields.CASES: ts_with_url_expected}
    )
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)

    one_region = dataset_out.get_one_region(test_helpers.DEFAULT_REGION)
    assert one_region.sources(CommonFields.ICU_BEDS) == [source_tag_prov_only]
    assert one_region.sources(CommonFields.CASES) == [source_tag_prov_with_url]


def test_make_source_tags_no_urls():
    # There was a bug where `./run.py data update` failed at the very end when no timeseries had
    # a source_url. This tests for it.
    ts_prov_only = TimeseriesLiteral(
        [0, 2, 4], annotation=[test_helpers.make_tag(date="2020-04-01"),], provenance="prov_only",
    )
    dataset_in = test_helpers.build_default_region_dataset({CommonFields.ICU_BEDS: ts_prov_only})

    dataset_out = timeseries.make_source_tags(dataset_in)

    source_tag_prov_only = taglib.Source("prov_only")
    ts_prov_only_expected = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[test_helpers.make_tag(date="2020-04-01"),],
        source=source_tag_prov_only,
    )
    dataset_expected = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only_expected}
    )
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)

    one_region = dataset_out.get_one_region(test_helpers.DEFAULT_REGION)
    assert one_region.sources(CommonFields.ICU_BEDS) == [source_tag_prov_only]


def test_make_source_url_tags():
    url_str = UrlStr("http://foo.com/1")

    source_tag_prov_only = taglib.Source("prov_only")
    ts_prov_only = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[test_helpers.make_tag(date="2020-04-01"),],
        source=source_tag_prov_only,
    )
    source_tag_prov_with_url = taglib.Source("prov_with_url", url=url_str)
    ts_with_url = TimeseriesLiteral([3, 5, 7], source=source_tag_prov_with_url,)
    dataset_in = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only, CommonFields.CASES: ts_with_url}
    )

    dataset_out = timeseries.make_source_url_tags(dataset_in)

    ts_with_url_expected = TimeseriesLiteral(
        [3, 5, 7], source=source_tag_prov_with_url, source_url=url_str
    )
    dataset_expected = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only, CommonFields.CASES: ts_with_url_expected}
    )
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_make_source_url_tags_no_source_tags():
    dataset_in = test_helpers.build_default_region_dataset({CommonFields.CASES: [1, 2, 3]})
    dataset_out = timeseries.make_source_url_tags(dataset_in)
    assert dataset_in == dataset_out


def test_make_source_url_tags_has_source_url():
    url_str = UrlStr("http://foo.com/1")
    dataset_in = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: TimeseriesLiteral([1, 2, 3], source_url=url_str)}
    )
    with pytest.raises(AssertionError):
        timeseries.make_source_url_tags(dataset_in)


def test_check_timeseries_structure_empty():
    timeseries._check_timeseries_wide_vars_structure(
        timeseries.EMPTY_TIMESERIES_WIDE_VARIABLES_DF, bucketed=False
    )
    timeseries._check_timeseries_wide_vars_structure(
        timeseries.EMPTY_TIMESERIES_BUCKETED_WIDE_VARIABLES_DF, bucketed=True
    )


def test_make_and_pickle_demographic_data():
    location_id = test_helpers.DEFAULT_REGION.location_id
    date_0 = test_helpers.DEFAULT_START_DATE
    date_1 = pd.to_datetime(test_helpers.DEFAULT_START_DATE) + pd.to_timedelta(1, unit="day")
    m1 = FieldName("m1")
    age20s = DemographicBucket("age:20-29")
    age30s = DemographicBucket("age:30-39")
    all = DemographicBucket("all")

    ds = test_helpers.build_default_region_dataset(
        {m1: {age20s: [1, 2, 3], age30s: [5, 6, 7], all: [8, 9, None]}}
    )

    assert ds.timeseries_bucketed.at[(location_id, age30s, date_0), m1] == 5
    assert ds.timeseries_bucketed.at[(location_id, all, date_1), m1] == 9

    ds_unpickled = pickle.loads(pickle.dumps(ds))

    test_helpers.assert_dataset_like(ds, ds_unpickled)
