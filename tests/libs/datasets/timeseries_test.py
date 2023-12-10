import dataclasses
import datetime
import io
import pathlib
import pickle
import warnings

import pytest
import pandas as pd
import numpy as np
import structlog

from datapublic.common_fields import CommonFields
from datapublic.common_fields import DemographicBucket
from datapublic.common_fields import FieldName
from datapublic.common_fields import PdFields

from datapublic.common_test_helpers import to_dict

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


# NOTE (sean 2023-12-10): Ignore FutureWarnings due to pandas MultiIndex .loc deprecations.
@pytest.fixture(autouse=True)
def ignore_future_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)


def _make_dataset_pointer(tmpdir, filename: str = "somefile.csv") -> dataset_pointer.DatasetPointer:
    # The fixture passes in a py.path, which is not the type in DatasetPointer.
    path = pathlib.Path(tmpdir) / filename

    fake_git_summary = github_utils.GitSummary(sha="abcdef", branch="main", is_dirty=True)

    return dataset_pointer.DatasetPointer(
        dataset_type=dataset_pointer.DatasetType.MULTI_REGION,
        path=path,
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
    # TODO(tom): Replace csv with test_helpers builders and uncomment assert in add_fips_static_df
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
    assert region_97111.date_indexed.at[pd.to_datetime("2020-04-02"), "m1"] == 2
    assert region_97111.latest["c1"] == 3
    assert multiregion.get_one_region(Region.from_fips("01")).latest["c2"] == 123.4

    csv_path = tmp_path / "multiregion.csv"
    multiregion.to_csv(csv_path)
    multiregion_loaded = timeseries.MultiRegionDataset.from_csv(csv_path)
    region_97111 = multiregion_loaded.get_one_region(Region.from_fips("97111"))
    assert region_97111.date_indexed.at[pd.to_datetime("2020-04-02"), "m1"] == 2
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
        pd.to_datetime("2020-04-01"): {
            "m2": 10,
            "location_id": "iso1:us#fips:97222",
        }
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
        Region.from_fips("97111"), pd.DataFrame([bar_county_row]), {}, pd.DataFrame([])
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
            Region.from_fips("97222"),
            pd.DataFrame([bar_county_row, foo_county_row]),
            {},
            pd.DataFrame([]),
        )

    with structlog.testing.capture_logs() as logs:
        ts = timeseries.OneRegionTimeseriesDataset(
            Region.from_fips("97111"),
            pd.DataFrame([], columns="location_id county aggregate_level date m1 m2".split()),
            {},
            pd.DataFrame([]),
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
            region_97111: {m1: TimeseriesLiteral([1, 2, None], provenance="src11")},
            region_97222: {
                m1: TimeseriesLiteral([None, None, 3], provenance="src21"),
                m2: TimeseriesLiteral([10, None, 30], provenance="src22"),
            },
            region_03: {
                m1: TimeseriesLiteral([None, None, 4], provenance="src31"),
                m2: TimeseriesLiteral([None, None, 40], provenance="src32"),
            },
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
                [0, 2, 4],
                annotation=[tag1, tag2],
                provenance=["prov1", "prov2"],
            ),
            CommonFields.CASES: [100, 200, 300],
        }
    )

    assert set(one_region.annotations_all_bucket(CommonFields.ICU_BEDS)) == {tag1, tag2}
    assert sorted(one_region.provenance[CommonFields.ICU_BEDS]) == ["prov1", "prov2"]


def test_add_aggregate_level():
    ts_df = read_csv_and_index_fips_date("fips,date,m1,m2\n" "36061,2020-04-02,2,\n").reset_index()
    multiregion = timeseries.MultiRegionDataset.from_fips_timeseries_df(ts_df)
    assert multiregion.geo_data.aggregate_level.to_list() == ["county"]


def test_fips_not_in_geo_data_csv_raises():
    df = test_helpers.read_csv_str(
        "       location_id,       date,       cases\n"
        "iso1:us#fips:06010, 2020-04-01,         100\n",
        skip_spaces=True,
    )

    with pytest.raises(AssertionError):
        timeseries.MultiRegionDataset.from_timeseries_df(df)


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
            "iso1:us#cbsa:20300,2020-04-03,4\n"
            "iso1:us#cbsa:10100,,3\n"
            "iso1:us#cbsa:20300,,4\n"
        )
    ).add_provenance_csv(
        io.StringIO("location_id,variable,provenance\n" "iso1:us#cbsa:20300,m1,prov20200m2\n")
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
            "iso1:us#cbsa:20300,2020-04-03,,,,4\n"
            "iso1:us#cbsa:10100,,,,,3\n"
            "iso1:us#cbsa:20300,,,,,4\n"
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
            "iso1:us#cbsa:20300,m1,prov20200m2\n"
        )
    )
    test_helpers.assert_dataset_like(ts_merged_1, ts_expected)


def test_append_regions_with_buckets():
    region_cbsa = Region.from_cbsa_code("10100")
    region_la = Region.from_fips("06037")
    region_sf = Region.from_fips("06075")
    m1 = FieldName("m1")
    m2 = FieldName("m2")
    age_40s = DemographicBucket("age:40-49")
    data_county = {
        region_la: {
            m1: {
                age_40s: TimeseriesLiteral([1, 2], annotation=[test_helpers.make_tag()]),
                DemographicBucket.ALL: [2, 3],
            }
        },
        region_sf: {m1: [3, 4]},
    }
    data_cbsa = {region_cbsa: {m2: [5, 6]}}
    ds_county = test_helpers.build_dataset(data_county)
    ds_cbsa = test_helpers.build_dataset(data_cbsa)

    ds_out_1 = ds_county.append_regions(ds_cbsa)
    ds_out_2 = ds_cbsa.append_regions(ds_county)

    ds_expected = test_helpers.build_dataset({**data_cbsa, **data_county})

    test_helpers.assert_dataset_like(ds_out_1, ds_expected)
    test_helpers.assert_dataset_like(ds_out_2, ds_expected)


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
    """Test timeseries_long where all data has bucket `all`"""
    region_cbsa = Region.from_cbsa_code("10100")
    region_county = Region.from_fips("97111")
    ds = test_helpers.build_dataset(
        {
            region_county: {FieldName("m1"): [2, None, 4]},
            region_cbsa: {FieldName("m2"): [2, 3, None]},
        },
        start_date="2020-04-02",
    )

    expected = test_helpers.read_csv_str(
        "       location_id,      date,variable,value\n"
        "iso1:us#cbsa:10100,2020-04-02,      m2,    2\n"
        "iso1:us#cbsa:10100,2020-04-03,      m2,    3\n"
        "iso1:us#fips:97111,2020-04-02,      m1,    2\n"
        "iso1:us#fips:97111,2020-04-04,      m1,    4\n",
        skip_spaces=True,
        dtype={"value": float},
    )
    long_series = ds.timeseries_bucketed_long
    assert long_series.index.names == [
        CommonFields.LOCATION_ID,
        PdFields.DEMOGRAPHIC_BUCKET,
        CommonFields.DATE,
        PdFields.VARIABLE,
    ]
    assert long_series.name == PdFields.VALUE
    long_df = long_series.xs("all", level=PdFields.DEMOGRAPHIC_BUCKET).reset_index()
    pd.testing.assert_frame_equal(long_df, expected, check_like=True)


def test_timeseries_bucketed_long():
    region_cbsa = Region.from_cbsa_code("10100")
    region_county = Region.from_fips("97111")
    bucket_age_0 = DemographicBucket("age:0-9")
    bucket_age_10 = DemographicBucket("age:10-19")
    bucket_all = DemographicBucket("all")
    ds = test_helpers.build_dataset(
        {
            region_county: {
                FieldName("m1"): {
                    bucket_age_0: [4, 5, 6],
                    bucket_age_10: [None, None, 7],
                    bucket_all: [2, None, 4],
                }
            },
            region_cbsa: {FieldName("m2"): [2, 3, None]},
        },
        start_date="2020-04-02",
    )

    expected = test_helpers.read_csv_str(
        "       location_id,demographic_bucket,      date,variable,value\n"
        "iso1:us#cbsa:10100,               all,2020-04-02,      m2,    2\n"
        "iso1:us#cbsa:10100,               all,2020-04-03,      m2,    3\n"
        "iso1:us#fips:97111,           age:0-9,2020-04-02,      m1,    4\n"
        "iso1:us#fips:97111,           age:0-9,2020-04-03,      m1,    5\n"
        "iso1:us#fips:97111,           age:0-9,2020-04-04,      m1,    6\n"
        "iso1:us#fips:97111,         age:10-19,2020-04-04,      m1,    7\n"
        "iso1:us#fips:97111,               all,2020-04-02,      m1,    2\n"
        "iso1:us#fips:97111,               all,2020-04-04,      m1,    4\n",
        skip_spaces=True,
        dtype={"value": float},
    )
    long_series = ds.timeseries_bucketed_long
    assert long_series.index.names == [
        CommonFields.LOCATION_ID,
        PdFields.DEMOGRAPHIC_BUCKET,
        CommonFields.DATE,
        PdFields.VARIABLE,
    ]
    assert long_series.name == PdFields.VALUE
    pd.testing.assert_frame_equal(long_series.reset_index(), expected, check_like=True)


def test_timeseries_distribution_long():
    bucket_age_0 = DemographicBucket("age:0-9")
    bucket_age_10 = DemographicBucket("age:10-19")
    bucket_all = DemographicBucket("all")
    bucket_blueman = DemographicBucket("color;gender:blue;man")
    ds = test_helpers.build_default_region_dataset(
        {
            FieldName("m1"): {
                bucket_age_0: [1, 2, 3],
                bucket_age_10: [None, None, 4],
                bucket_all: [5, None, 6],
                bucket_blueman: [7, None, None],
            }
        }
    )

    long_series = ds.timeseries_distribution_long
    assert long_series.name == PdFields.VALUE
    assert long_series.index.names == [
        CommonFields.LOCATION_ID,
        PdFields.DEMOGRAPHIC_BUCKET,
        CommonFields.DATE,
        PdFields.DISTRIBUTION,
        PdFields.VARIABLE,
    ]
    expected = test_helpers.read_csv_str(
        "       location_id,   demographic_bucket,      date,distribution,variable,value\n"
        "iso1:us#fips:97222,              age:0-9,2020-04-01,         age,      m1,    1\n"
        "iso1:us#fips:97222,              age:0-9,2020-04-02,         age,      m1,    2\n"
        "iso1:us#fips:97222,              age:0-9,2020-04-03,         age,      m1,    3\n"
        "iso1:us#fips:97222,            age:10-19,2020-04-03,         age,      m1,    4\n"
        "iso1:us#fips:97222,                  all,2020-04-01,         all,      m1,    5\n"
        "iso1:us#fips:97222,                  all,2020-04-03,         all,      m1,    6\n"
        "iso1:us#fips:97222,color;gender:blue;man,2020-04-01,color;gender,      m1,    7\n",
        skip_spaces=True,
        dtype={"value": float},
    )
    pd.testing.assert_frame_equal(long_series.reset_index(), expected, check_like=True)


def test_timeseries_wide_dates():
    region_cbsa = Region.from_cbsa_code("10100")
    region_fips = Region.from_fips("97111")
    m1 = FieldName("m1")
    m2 = FieldName("m2")
    ds = test_helpers.build_dataset(
        {region_cbsa: {m2: [2, 3]}, region_fips: {m1: [2, None, 4]}},
        static_by_region_then_field_name={region_fips: {CommonFields.COUNTY: "Bar County", m1: 4}},
        start_date="2020-04-02",
    )

    # TODO(tom): Delete this test of _timeseries_not_bucketed_wide_dates which is no longer
    #  accessed from outside timeseries when there are other tests for from_timeseries_wide_dates_df
    ds_wide = ds._timeseries_not_bucketed_wide_dates
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

    assert ds.get_timeseries_not_bucketed_wide_dates(m1).loc[region_fips.location_id, :].replace(
        {np.nan: None}
    ).to_list() == [2, None, 4]
    assert ds.get_timeseries_bucketed_wide_dates(m1).loc[
        (region_fips.location_id, DemographicBucket.ALL), :
    ].replace({np.nan: None}).to_list() == [2, None, 4]


def test_timeseries_wide_dates_empty():
    m1 = FieldName("m1")
    ds = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,county,aggregate_level,m1,m2\n"
            "iso1:us#cbsa:10100,,,,,3\n"
            "iso1:us#fips:97111,,Bar County,county,4,\n"
        )
    )

    assert ds.get_timeseries_not_bucketed_wide_dates(m1).empty
    assert ds.get_timeseries_bucketed_wide_dates(m1).empty
    assert ds.get_timeseries_not_bucketed_wide_dates(CommonFields.CASES).empty
    assert ds.get_timeseries_bucketed_wide_dates(CommonFields.CASES).empty


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


def test_write_read_wide_with_buckets(tmpdir):
    pointer = _make_dataset_pointer(tmpdir)

    all_bucket = DemographicBucket("all")
    age_20s = DemographicBucket("age:20-29")
    age_30s = DemographicBucket("age:30-39")
    region_as = Region.from_state("AS")
    region_sf = Region.from_fips("06075")
    metrics_as = {
        CommonFields.ICU_BEDS: TimeseriesLiteral(
            [0, 2, 4],
            annotation=[
                test_helpers.make_tag(date="2020-04-01"),
                test_helpers.make_tag(TagType.ZSCORE_OUTLIER, date="2020-04-02"),
            ],
        ),
        CommonFields.CASES: [100, 200, 300],
    }
    metrics_sf = {
        CommonFields.CASES: {
            age_20s: TimeseriesLiteral([3, 4, 5], source=taglib.Source(type="MySource")),
            age_30s: [4, 5, 6],
            all_bucket: [1, 2, 3],
        }
    }
    dataset_in = test_helpers.build_dataset({region_as: metrics_as, region_sf: metrics_sf})

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
    assert one_region_tx.annotations_all_bucket(CommonFields.CASES) == [tag1]
    one_region_sf = dataset_tx_and_sf.get_one_region(region_sf)
    assert one_region_sf.annotations_all_bucket(CommonFields.CASES) == [
        tag2a,
        tag2b,
    ]
    assert set(one_region_sf.sources_all_bucket(CommonFields.CASES)) == set()

    assert {
        region: one_region_dataset.annotations_all_bucket(CommonFields.CASES)
        for region, one_region_dataset in dataset_tx_and_sf.iter_one_regions()
    } == {
        region_sf: [tag2a, tag2b],
        region_tx: [tag1],
    }


def test_one_region_empty_annotations():
    one_region = test_helpers.build_one_region_dataset({CommonFields.CASES: [100, 200, 300]})

    assert one_region.annotations_all_bucket(CommonFields.CASES) == []
    assert one_region.source_url == {}
    assert one_region.provenance == {}
    assert set(one_region.sources_all_bucket(CommonFields.ICU_BEDS)) == set()
    assert set(one_region.sources_all_bucket(CommonFields.CASES)) == set()


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
        (CommonFields.CASES, DemographicBucket.ALL, tag1.tag_type, tag1),
        (
            CommonFields.ICU_BEDS,
            DemographicBucket.ALL,
            "provenance",
            taglib.ProvenanceTag(source="prov1"),
        ),
        (CommonFields.DEATHS, DemographicBucket.ALL, tag2a.tag_type, tag2a),
        (CommonFields.DEATHS, DemographicBucket.ALL, tag2b.tag_type, tag2b),
    }


def test_one_region_tag_objects_series_empty():
    one_region = test_helpers.build_one_region_dataset({CommonFields.CASES: [1, 2, 3]})
    assert one_region.tag.empty
    assert isinstance(one_region.tag_objects_series, pd.Series)
    assert one_region.tag_objects_series.empty


def test_timeseries_tag_objects_series():
    values = [100, 200]
    tag1 = test_helpers.make_tag(TagType.ZSCORE_OUTLIER, date="2020-04-01")
    tag2a = test_helpers.make_tag(date="2020-04-02")
    tag2b = test_helpers.make_tag(date="2020-04-03")
    url_str = UrlStr("http://foo.com/1")
    source_obj = taglib.Source("source_with_url", url=url_str)

    ds = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: TimeseriesLiteral(values, annotation=[tag1]),
            CommonFields.ICU_BEDS: TimeseriesLiteral(values, source=source_obj),
            CommonFields.DEATHS: TimeseriesLiteral(values, annotation=[tag2a, tag2b]),
            CommonFields.TOTAL_TESTS: values,
        }
    )

    assert isinstance(ds.tag_objects_series, pd.Series)
    assert ds.tag.index.equals(ds.tag_objects_series.index)
    location_id = test_helpers.DEFAULT_REGION.location_id
    assert set(ds.tag_objects_series.reset_index().itertuples(index=False)) == {
        (location_id, CommonFields.CASES, DemographicBucket.ALL, tag1.tag_type, tag1),
        (location_id, CommonFields.ICU_BEDS, DemographicBucket.ALL, TagType.SOURCE, source_obj),
        (location_id, CommonFields.DEATHS, DemographicBucket.ALL, tag2a.tag_type, tag2a),
        (location_id, CommonFields.DEATHS, DemographicBucket.ALL, tag2b.tag_type, tag2b),
    }


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
        "aggregate_level": "cbsa",
        "county": None,
        "country": "USA",
        "fips": "10100",
        "state": None,
        "m1": 10,  # Derived from timeseries
        "m2": 4,  # Explicitly in recent values
    }
    region_97111 = dataset.get_one_region(Region.from_fips("97111"))
    assert region_97111.latest == {
        "aggregate_level": "county",
        "county": "Bar County",
        "country": "USA",
        "fips": "97111",
        "state": "ZZ",
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

    ts_joined = ts_2.join_columns(ts_1)
    test_helpers.assert_dataset_like(ts_joined, ts_expected, drop_na_latest=True)

    with pytest.raises(ValueError):
        # Raises because the same column is in both datasets
        ts_2.join_columns(ts_2)

    # geo attributes, such as aggregation level and county name, generally appear in geo-data.csv
    # instead of MultiRegionDataset so they don't need special handling in join_columns.


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


def test_join_columns_with_buckets():
    m1 = FieldName("m1")
    m2 = FieldName("m2")
    age20s = DemographicBucket("age:20-29")

    m1_data = {m1: {age20s: [1, 2, 3]}}
    ds_1 = test_helpers.build_default_region_dataset(m1_data)
    m2_data = {m2: {age20s: [4, 5, 6], DemographicBucket.ALL: [7, 8, 9]}}
    ds_2 = test_helpers.build_default_region_dataset(m2_data)

    with pytest.raises(ValueError):
        ds_1.join_columns(ds_1)

    ds_expected = test_helpers.build_default_region_dataset({**m1_data, **m2_data})

    ds_joined = ds_1.join_columns(ds_2)
    test_helpers.assert_dataset_like(ds_joined, ds_expected)


def test_join_columns_with_static():
    m1 = FieldName("m1")
    m2 = FieldName("m2")

    ds_1 = test_helpers.build_default_region_dataset({}, static={m1: 1})
    ds_2 = test_helpers.build_default_region_dataset({}, static={m2: 2})

    with pytest.raises(ValueError):
        ds_1.join_columns(ds_1)

    ds_expected = test_helpers.build_default_region_dataset({}, static={m1: 1, m2: 2})

    ds_joined = ds_1.join_columns(ds_2)
    test_helpers.assert_dataset_like(ds_joined, ds_expected)

    ds_joined = ds_2.join_columns(ds_1)
    test_helpers.assert_dataset_like(ds_joined, ds_expected)


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
    cbsa_with_pop = Region.from_cbsa_code("10100")
    fips_with_pop = Region.from_fips("97111")
    cbsa_without_pop = Region.from_cbsa_code("20300")
    fips_without_pop = Region.from_fips("97222")
    m1 = FieldName("m1")
    regions_with_pop = [cbsa_with_pop, fips_with_pop]
    all_regions = regions_with_pop + [cbsa_without_pop, fips_without_pop]
    static_populations = {r: {CommonFields.POPULATION: 80_000} for r in regions_with_pop}
    ts_in = test_helpers.build_dataset(
        {r: {m1: [1]} for r in all_regions},
        static_by_region_then_field_name=static_populations,
    )
    ts_expected = test_helpers.build_dataset(
        {r: {m1: [1]} for r in regions_with_pop},
        static_by_region_then_field_name=static_populations,
    )
    with structlog.testing.capture_logs() as logs:
        ts_out = timeseries.drop_regions_without_population(
            ts_in, [fips_without_pop.location_id], structlog.get_logger()
        )
    test_helpers.assert_dataset_like(ts_out, ts_expected)

    assert [l["event"] for l in logs] == ["Dropping unexpected regions without populaton"]
    assert [l["location_ids"] for l in logs] == [[cbsa_without_pop.location_id]]


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
    tag_df = test_helpers.make_tag_df(
        region_sf, CommonFields.CASES, DemographicBucket.ALL, [tag_sf_cases]
    )
    dataset_out = dataset_in.append_tag_df(tag_df)
    metrics_sf[CommonFields.CASES] = TimeseriesLiteral(cases_values, annotation=[tag_sf_cases])
    dataset_expected = test_helpers.build_dataset({region_sf: metrics_sf})

    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_add_provenance_all_with_tags():
    """Checks that add_provenance_all (and add_provenance_series that it calls) fails when tags
    already exist."""
    region = Region.from_state("TX")
    cases_values = [100, 200, 300, 400]
    timeseries = TimeseriesLiteral(cases_values, annotation=[(test_helpers.make_tag())])
    dataset_in = test_helpers.build_dataset({region: {CommonFields.CASES: timeseries}})

    with pytest.raises(NotImplementedError):
        dataset_in.add_provenance_all("prov_prov")


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


def test_drop_na_columns():
    tag = test_helpers.make_tag()
    timeseries_real = {
        CommonFields.CASES: TimeseriesLiteral([1, 2], annotation=[tag]),
    }
    static_real = {CommonFields.STAFFED_BEDS: 3}
    ds = test_helpers.build_default_region_dataset(
        # Adds CASES with real values, which won't be dropped, and a tag for DEATHS, that will be
        # dropped.
        {**timeseries_real, CommonFields.DEATHS: TimeseriesLiteral([], annotation=[tag])},
        static=static_real,
    )
    # The test_helper functions don't do a good job of creating fields that are all NA so the
    # following inserts time series DEATHS and static ICU_BEDS, then asserts that they were
    # inserted.
    timeseries_bucketed_with_na = ds.timeseries_bucketed.copy()
    timeseries_bucketed_with_na.loc[:, CommonFields.DEATHS] = np.nan
    static_with_na = ds.static.copy()
    static_with_na.loc[:, CommonFields.ICU_BEDS] = np.nan
    ds = dataclasses.replace(
        ds, timeseries_bucketed=timeseries_bucketed_with_na, static=static_with_na
    )
    assert CommonFields.DEATHS in ds.timeseries_bucketed.columns
    assert CommonFields.ICU_BEDS in ds.static.columns

    dataset_out = ds.drop_na_columns()

    dataset_expected = test_helpers.build_default_region_dataset(
        timeseries_real, static=static_real
    )
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_drop_na_columns_no_tags():
    timeseries_real = {CommonFields.CASES: [1, 2]}
    tag = test_helpers.make_tag()
    ds = test_helpers.build_default_region_dataset(
        # Add a tag for DEATHS, that will be dropped.
        {**timeseries_real, CommonFields.DEATHS: TimeseriesLiteral([], annotation=[tag])}
    )
    # The test_helper functions don't do a good job of creating fields that are all NA so the
    # following inserts time series DEATHS and static ICU_BEDS, then asserts that they were
    # inserted.
    timeseries_bucketed_with_na = ds.timeseries_bucketed.copy()
    timeseries_bucketed_with_na.loc[:, CommonFields.DEATHS] = np.nan
    ds = dataclasses.replace(ds, timeseries_bucketed=timeseries_bucketed_with_na)
    assert CommonFields.DEATHS in ds.timeseries_bucketed.columns

    dataset_out = ds.drop_na_columns()

    dataset_expected = test_helpers.build_default_region_dataset(timeseries_real)
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)


def test_drop_column_with_tags_and_bucket():
    age_40s = DemographicBucket("age:40-49")
    ts_lit = TimeseriesLiteral([10, 20, 30], annotation=[test_helpers.make_tag()])
    data_cases = {CommonFields.CASES: {age_40s: ts_lit, DemographicBucket.ALL: ts_lit}}
    data_deaths = {CommonFields.DEATHS: {age_40s: ts_lit}}

    dataset_in = test_helpers.build_default_region_dataset({**data_cases, **data_deaths})
    assert len(dataset_in.tag) == 3

    dataset_out = dataset_in.drop_column_if_present(CommonFields.DEATHS)

    assert len(dataset_out.tag) == 2
    dataset_expected = test_helpers.build_default_region_dataset({**data_cases})
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
    dataset = timeseries.MultiRegionDataset.from_timeseries_df(
        pd.DataFrame([], columns=[CommonFields.LOCATION_ID, CommonFields.DATE])
    )
    with pytest.raises(timeseries.RegionLatestNotFound):
        dataset.get_one_region(Region.from_fips("01001"))


def test_timeseries_empty_static_not_empty():
    # Check that empty timeseries does not prevent static data working as expected.
    dataset = timeseries.MultiRegionDataset.from_timeseries_df(
        pd.DataFrame([], columns=[CommonFields.LOCATION_ID, CommonFields.DATE])
    ).add_static_values(pd.DataFrame([{"location_id": "iso1:us#fips:97111", "m1": 1234}]))
    assert dataset.get_one_region(Region.from_fips("97111")).latest["m1"] == 1234


def test_from_timeseries_df_fips_location_id_mismatch():
    df = test_helpers.read_csv_str(
        "                  location_id, fips,      date,m1\n"
        "iso1:us#iso2:us-tx#fips:48197,48201,2020-04-02, 2\n"
        "iso1:us#iso2:us-tx#fips:48201,48201,2020-04-02, 2\n",
        skip_spaces=True,
    )
    with pytest.warns(timeseries.ExtraColumnWarning, match="48201"):
        timeseries.MultiRegionDataset.from_timeseries_df(df)


def test_from_timeseries_df_no_fips_no_warning():
    df = test_helpers.read_csv_str(
        "            location_id, fips,      date,m1\n"
        "                iso1:us,     ,2020-04-02, 2\n",
        skip_spaces=True,
    )
    timeseries.MultiRegionDataset.from_timeseries_df(df)


def test_from_timeseries_df_fips_state_mismatch():
    df = test_helpers.read_csv_str(
        "                  location_id,state,      date,m1\n"
        "iso1:us#iso2:us-tx#fips:48197,   TX,2020-04-02, 2\n"
        "iso1:us#iso2:us-tx#fips:48201,   IL,2020-04-02, 2\n",
        skip_spaces=True,
    )
    with pytest.warns(timeseries.ExtraColumnWarning, match="48201"):
        timeseries.MultiRegionDataset.from_timeseries_df(df)


def test_from_timeseries_df_bad_level():
    df = test_helpers.read_csv_str(
        "                  location_id, aggregate_level,      date,m1\n"
        "iso1:us#iso2:us-tx#fips:48201,          county,2020-04-02, 2\n"
        "iso1:us#iso2:us-tx#fips:48197,           state,2020-04-02, 2\n"
        "           iso1:us#iso2:us-tx,           state,2020-04-02, 2\n",
        skip_spaces=True,
    )
    with pytest.warns(timeseries.ExtraColumnWarning, match="48197"):
        timeseries.MultiRegionDataset.from_timeseries_df(df)


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
    ts2a = TimeseriesLiteral(
        [1, 3, 5],
        annotation=[test_helpers.make_tag(date="2020-04-01")],
    )
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
    expected = test_helpers.read_csv_str(
        "       location_id,variable,demographic_bucket,2020-04-02,2020-04-01\n"
        "iso1:us#iso2:us-az,      m1,               all,        12,         8\n"
        "iso1:us#iso2:us-az,      m2,               all,        40,        20\n"
        "iso1:us#iso2:us-tx,      m1,               all,         4,         4\n"
        "iso1:us#iso2:us-tx,      m2,               all,         4,         2\n",
        skip_spaces=True,
    ).set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET])
    pd.testing.assert_frame_equal(rows, expected, check_dtype=False, check_exact=False)


def test_multi_region_dataset_get_subset():
    region_us = Region.from_iso1("us")
    region_tx = Region.from_state("TX")
    region_county = Region.from_fips("97222")
    region_cbsa = Region.from_cbsa_code("10100")
    m1 = FieldName("m1")
    m2 = FieldName("m2")
    ds = test_helpers.build_dataset(
        {
            region_us: {m1: [100], m2: [200]},
            region_tx: {m1: [4], m2: [2]},
            region_county: {m1: [1], m2: [2]},
            region_cbsa: {m1: [1], m2: [2], CommonFields.POPULATION: [20_000]},
        },
        static_by_region_then_field_name={
            region_us: {CommonFields.POPULATION: 10_000},
            region_tx: {CommonFields.POPULATION: 5_000},
            region_county: {CommonFields.POPULATION: 1_000},
        },
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


def test_multi_region_dataset_get_subset_with_buckets():
    # Make some regions at different levels
    region_us = Region.from_iso1("us")
    region_tx = Region.from_state("TX")
    region_la = Region.from_fips("06037")
    age_40s = DemographicBucket("age:40-49")
    data_us = {region_us: {CommonFields.CASES: [100, 200]}}
    data_tx = {region_tx: {CommonFields.CASES: [10, 20]}}
    data_la = {region_la: {CommonFields.CASES: {DemographicBucket.ALL: [5, 10], age_40s: [1, 2]}}}
    ds = test_helpers.build_dataset({**data_us, **data_tx, **data_la})

    ds_expected = test_helpers.build_dataset({**data_us, **data_la})
    test_helpers.assert_dataset_like(ds.get_regions_subset([region_us, region_la]), ds_expected)
    test_helpers.assert_dataset_like(ds.partition_by_region(exclude=[region_tx])[0], ds_expected)


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
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
        ],
        provenance="prov_only",
    )
    ts_with_url = TimeseriesLiteral([3, 5, 7], provenance="prov_with_url", source_url=url_str)
    dataset_in = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only, CommonFields.CASES: ts_with_url}
    )

    dataset_out = timeseries.make_source_tags(dataset_in)

    source_tag_prov_only = taglib.Source("prov_only")
    ts_prov_only_expected = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
        ],
        source=source_tag_prov_only,
    )
    source_tag_prov_with_url = taglib.Source("prov_with_url", url=url_str)
    ts_with_url_expected = TimeseriesLiteral(
        [3, 5, 7],
        source=source_tag_prov_with_url,
    )
    dataset_expected = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only_expected, CommonFields.CASES: ts_with_url_expected}
    )
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)

    one_region = dataset_out.get_one_region(test_helpers.DEFAULT_REGION)
    assert one_region.sources_all_bucket(CommonFields.ICU_BEDS) == [source_tag_prov_only]
    assert one_region.sources_all_bucket(CommonFields.CASES) == [source_tag_prov_with_url]


def test_make_source_tags_no_urls():
    # There was a bug where `./run.py data update` failed at the very end when no timeseries had
    # a source_url. This tests for it.
    ts_prov_only = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
        ],
        provenance="prov_only",
    )
    dataset_in = test_helpers.build_default_region_dataset({CommonFields.ICU_BEDS: ts_prov_only})

    dataset_out = timeseries.make_source_tags(dataset_in)

    source_tag_prov_only = taglib.Source("prov_only")
    ts_prov_only_expected = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
        ],
        source=source_tag_prov_only,
    )
    dataset_expected = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: ts_prov_only_expected}
    )
    test_helpers.assert_dataset_like(dataset_out, dataset_expected)

    one_region = dataset_out.get_one_region(test_helpers.DEFAULT_REGION)
    assert one_region.sources_all_bucket(CommonFields.ICU_BEDS) == [source_tag_prov_only]


def test_make_source_url_tags():
    url_str = UrlStr("http://foo.com/1")

    source_tag_prov_only = taglib.Source("prov_only")
    ts_prov_only = TimeseriesLiteral(
        [0, 2, 4],
        annotation=[
            test_helpers.make_tag(date="2020-04-01"),
        ],
        source=source_tag_prov_only,
    )
    source_tag_prov_with_url = taglib.Source("prov_with_url", url=url_str)
    ts_with_url = TimeseriesLiteral(
        [3, 5, 7],
        source=source_tag_prov_with_url,
    )
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


def test_combine_demographic_data_basic():
    m1 = FieldName("m1")
    age20s = DemographicBucket("age:20-29")
    age30s = DemographicBucket("age:30-39")
    age40s = DemographicBucket("age:40-49")
    ds1 = test_helpers.build_default_region_dataset(
        {
            m1: {
                age20s: [21, 22, 23],
                age30s: [31, 32, 33],
            }
        }
    )
    ds2 = test_helpers.build_default_region_dataset(
        {
            m1: {
                age30s: [32, 33, 34],
                age40s: [42, 43, 44],
            }
        }
    )

    combined = timeseries.combined_datasets({m1: [ds1, ds2]}, {})
    test_helpers.assert_dataset_like(combined, ds1)

    combined = timeseries.combined_datasets({m1: [ds2, ds1]}, {})
    test_helpers.assert_dataset_like(combined, ds2)


def test_combine_demographic_data_multiple_distributions():
    """All time-series within a variable are treated as a unit when combining"""
    m1 = FieldName("m1")
    m2 = FieldName("m2")
    all = DemographicBucket("all")
    age_20s = DemographicBucket("age:20-29")
    age_30s = DemographicBucket("age:30-39")
    region_ak = Region.from_state("AK")
    region_ca = Region.from_state("CA")

    ds1 = test_helpers.build_dataset(
        {
            region_ak: {m1: {all: TimeseriesLiteral([1.1, 2.1], provenance="ds1_ak_m1_all")}},
            region_ca: {m1: {age_20s: TimeseriesLiteral([2.1, 3.1], provenance="ds1_ca_m1_20s")}},
        }
    )

    ds2 = test_helpers.build_dataset(
        {
            region_ak: {m1: {all: TimeseriesLiteral([1, 2], provenance="ds2_ak_m1_all")}},
            region_ca: {
                m1: {
                    age_30s: TimeseriesLiteral([3, 4], provenance="ds2_ca_m1_30s"),
                    all: TimeseriesLiteral([5, 6], provenance="ds2_ca_m1_all"),
                },
                m2: {age_30s: TimeseriesLiteral([6, 7], provenance="ds2_ca_m2_30s")},
            },
        }
    )

    combined = timeseries.combined_datasets({m1: [ds1, ds2], m2: [ds1, ds2]}, {})

    ds_expected = test_helpers.build_dataset(
        {
            region_ak: {m1: {all: TimeseriesLiteral([1.1, 2.1], provenance="ds1_ak_m1_all")}},
            region_ca: {
                m1: {
                    age_20s: TimeseriesLiteral([2.1, 3.1], provenance="ds1_ca_m1_20s"),
                    all: TimeseriesLiteral([5, 6], provenance="ds2_ca_m1_all"),
                },
                m2: {age_30s: TimeseriesLiteral([6, 7], provenance="ds2_ca_m2_30s")},
            },
        }
    )
    test_helpers.assert_dataset_like(combined, ds_expected)


def test_bucketed_latest_missing_location_id(nyc_region: Region):
    dataset = test_helpers.build_default_region_dataset({CommonFields.CASES: [1, 2, 3]})
    # nyc_region = Region.from_fips("97222")
    output = dataset._bucketed_latest_for_location_id(nyc_region.location_id)
    expected = pd.DataFrame(
        [],
        index=pd.MultiIndex.from_tuples([], names=[PdFields.DEMOGRAPHIC_BUCKET]),
        columns=pd.Index([CommonFields.CASES], name="variable"),
        dtype="float",
    )
    pd.testing.assert_frame_equal(expected, output)


def test_bucketed_latest(nyc_region: Region):
    m1 = FieldName("m1")
    age20s = DemographicBucket("age:20-29")
    age30s = DemographicBucket("age:30-39")

    dataset = test_helpers.build_default_region_dataset(
        {
            m1: {
                age20s: [21, 22, 23],
                age30s: [31, 32, 33],
            }
        }
    )
    bucketed_latest = dataset._bucketed_latest_for_location_id(
        test_helpers.DEFAULT_REGION.location_id
    )
    expected = pd.DataFrame(
        [{"m1": 23}, {"m1": 33}],
        index=pd.Index([age20s, age30s], name=PdFields.DEMOGRAPHIC_BUCKET),
        columns=pd.Index([m1], name="variable"),
    )
    pd.testing.assert_frame_equal(bucketed_latest, expected)


def test_one_region_demographic_distributions():
    m1 = FieldName("m1")
    age20s = DemographicBucket("age:20-29")
    age30s = DemographicBucket("age:30-39")
    dataset = test_helpers.build_default_region_dataset(
        {m1: {age20s: [21, 22, 23], age30s: [31, 32, 33], DemographicBucket.ALL: [20, 21, 22]}}
    )
    one_region = dataset.get_one_region(test_helpers.DEFAULT_REGION)

    expected = {m1: {"age": {"20-29": 23, "30-39": 33}}}
    assert one_region.demographic_distributions_by_field == expected


def test_one_region_demographic_distributions_overlapping_buckets():
    m1 = FieldName("m1")
    m2 = FieldName("m2")
    age20s = DemographicBucket("age:20-29")
    age30s = DemographicBucket("age:30-39")
    # Presumably 25 to 29 is from a different age distribution as it overlaps with age bucket above.
    # Make sure that different age bucketing doesn't polute other variables.
    age25to29 = DemographicBucket("age:25-29")

    dataset = test_helpers.build_default_region_dataset(
        {
            m1: {age20s: [21, 22, 23], age30s: [31, 32, 33], DemographicBucket.ALL: [20, 21, 22]},
            m2: {DemographicBucket.ALL: [20, 21, 22], age25to29: [20, 21, 22]},
        },
    )
    one_region = dataset.get_one_region(test_helpers.DEFAULT_REGION)
    expected = {m1: {"age": {"20-29": 23, "30-39": 33}}, m2: {"age": {"25-29": 22}}}

    assert one_region.demographic_distributions_by_field == expected


def test_print_stats():
    all_bucket = DemographicBucket("all")
    age_20s = DemographicBucket("age:20-29")
    age_30s = DemographicBucket("age:30-39")

    test_helpers.build_default_region_dataset(
        {
            CommonFields.ICU_BEDS: TimeseriesLiteral(
                [0, 2, 4],
                annotation=[
                    test_helpers.make_tag(date="2020-04-01"),
                ],
            ),
            CommonFields.CASES: [100, 200, 300],
        }
    ).print_stats("DS1")

    test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {
                age_20s: TimeseriesLiteral([3, 4, 5], source=taglib.Source(type="MySource")),
                age_30s: [4, 5, 6],
                all_bucket: [1, 2, 3],
            }
        }
    ).print_stats("DS2")


def test_static_and_geo_data():
    region_chi = Region.from_fips("17031")
    ds = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: [0]}, static={CommonFields.POPULATION: 5}, region=region_chi
    )
    assert ds.static_and_geo_data.loc[region_chi.location_id, CommonFields.COUNTY] == "Cook County"
    assert ds.static_and_geo_data.loc[region_chi.location_id, CommonFields.POPULATION] == 5


def test_add_tag_all_bucket():
    region_tx = Region.from_state("TX")
    region_la = Region.from_fips("06037")
    age_40s = DemographicBucket("age:40-49")
    data_tx = {region_tx: {CommonFields.CASES: [10, 20]}}
    data_la = {region_la: {CommonFields.CASES: {DemographicBucket.ALL: [5, 10], age_40s: [1, 2]}}}

    tag = test_helpers.make_tag(date="2020-04-01")
    ds = test_helpers.build_dataset({**data_tx, **data_la}).add_tag_all_bucket(tag)

    expected_tx = {region_tx: {CommonFields.CASES: TimeseriesLiteral([10, 20], annotation=[tag])}}
    expected_la = {
        region_la: {
            CommonFields.CASES: {
                DemographicBucket.ALL: TimeseriesLiteral([5, 10], annotation=[tag]),
                age_40s: [1, 2],
            }
        }
    }
    ds_expected = test_helpers.build_dataset({**expected_tx, **expected_la})
    test_helpers.assert_dataset_like(ds, ds_expected)


def test_add_tag_without_timeseries(tmpdir):
    """Create a dataset with a tag for a timeseries that doesn't exist."""
    pointer = _make_dataset_pointer(tmpdir)

    region_tx = Region.from_state("TX")
    region_la = Region.from_fips("06037")
    data_tx = {region_tx: {CommonFields.CASES: [10, 20]}}

    tag_collection = taglib.TagCollection()
    tag = test_helpers.make_tag(date="2020-04-01")
    tag_collection.add(
        tag,
        location_id=region_la.location_id,
        variable=CommonFields.CASES,
        bucket=DemographicBucket.ALL,
    )

    dataset = test_helpers.build_dataset({**data_tx}).append_tag_df(tag_collection.as_dataframe())

    # Check that the tag was created for region_la, which doesn't have any timeseries data.
    assert set(
        dataset.tag_objects_series.xs(region_la.location_id, level=CommonFields.LOCATION_ID)
    ) == {tag}

    # Check that tag location_id are included in location_ids property.
    assert set(dataset.location_ids) == {region_la.location_id, region_tx.location_id}

    # Check that the tag still exists after writing and reading from disk.
    dataset.write_to_dataset_pointer(pointer)
    dataset_read = timeseries.MultiRegionDataset.read_from_pointer(pointer)
    test_helpers.assert_dataset_like(dataset, dataset_read)


def test_variables():
    # Make a dataset with CASES, DEATHS and ICU_BEDS each appearing in only one of timeseries,
    # static and tag data. This make sure variable names are merged from all three places.
    region_97111 = Region.from_fips("97111")
    tag_collection = taglib.TagCollection()
    tag_collection.add(
        test_helpers.make_tag(),
        location_id=region_97111.location_id,
        variable=CommonFields.DEATHS,
        bucket=DemographicBucket.ALL,
    )
    ds = test_helpers.build_dataset(
        {region_97111: {CommonFields.CASES: [1, 2, None]}},
        static_by_region_then_field_name={region_97111: {CommonFields.ICU_BEDS: 10}},
    ).append_tag_df(tag_collection.as_dataframe())
    assert set(ds.variables.to_list()) == {
        CommonFields.CASES,
        CommonFields.ICU_BEDS,
        CommonFields.DEATHS,
    }


def test_variables_empty():
    assert timeseries.MultiRegionDataset.new_without_timeseries().variables.to_list() == []


def test_static_long():
    region_cbsa = Region.from_cbsa_code("10100")
    region_fips = Region.from_fips("97111")
    m1 = FieldName("m1")
    ds = test_helpers.build_dataset(
        {},
        static_by_region_then_field_name={
            region_fips: {CommonFields.CAN_LOCATION_PAGE_URL: "http://can.do", m1: 4},
            region_cbsa: {CommonFields.CASES: 3},
        },
    )
    # Use loc[level0].at[level1] as work-around for
    # https://github.com/pandas-dev/pandas/issues/26989
    # TODO(tom): Change to `at[level0, level1]` after upgrading to Pandas >=1.1
    assert (
        ds.static_long.loc[region_fips.location_id].at[CommonFields.CAN_LOCATION_PAGE_URL]
        == "http://can.do"
    )
    assert ds.static_long.loc[region_fips.location_id].at[m1] == 4
    assert ds.static_long.loc[region_cbsa.location_id].at[CommonFields.CASES] == 3

    ds_empty_static = timeseries.MultiRegionDataset.new_without_timeseries()
    assert ds_empty_static.static_long.empty
    assert ds_empty_static.static_long.name == ds.static_long.name
    assert ds_empty_static.static_long.index.names == ds.static_long.index.names


def test_delta_timeseries_removed():
    # This tests time series being removed only, not tags or static values.
    region_tx = Region.from_state("TX")
    region_la = Region.from_fips("06037")
    age_40s = DemographicBucket("age:40-49")
    data_tx = {region_tx: {CommonFields.CASES: [10, 20]}}
    data_la_a = {region_la: {CommonFields.CASES: {DemographicBucket.ALL: [5, 10], age_40s: [1, 2]}}}

    ds_a = test_helpers.build_dataset({**data_tx, **data_la_a})

    data_la_b = {region_la: {CommonFields.CASES: {DemographicBucket.ALL: [5, 10]}}}
    ds_b = test_helpers.build_dataset({**data_tx, **data_la_b})

    delta = timeseries.MultiRegionDatasetDiff(old=ds_a, new=ds_b)
    ds_out = delta.timeseries_removed

    ds_expected = test_helpers.build_dataset({region_la: {CommonFields.CASES: {age_40s: [1, 2]}}})

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_drop_observations_after():
    age_40s = DemographicBucket("age:40-49")
    ds_in = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {DemographicBucket.ALL: [5, 10], age_40s: [1, 2, 3]},
            # Check that observation is dropped even when not a True value (ie 0).
            CommonFields.DEATHS: [0, 0, 0],
            # Check what happens when there are no real valued observations after dropping,
            # though the behaviour probably doesn't matter.
            CommonFields.ICU_BEDS: [None, None, 10],
        }
    )

    ds_out = timeseries.drop_observations(ds_in, after=datetime.date(2020, 4, 2))

    tag = test_helpers.make_tag(taglib.TagType.DROP_FUTURE_OBSERVATION, after="2020-04-02")
    ds_expected = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {
                DemographicBucket.ALL: [5, 10],
                age_40s: TimeseriesLiteral([1, 2], annotation=[tag]),
            },
            CommonFields.DEATHS: TimeseriesLiteral([0, 0], annotation=[tag]),
            CommonFields.ICU_BEDS: TimeseriesLiteral([], annotation=[tag]),
        }
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_pickle_test_dataset_size(tmp_path: pathlib.Path):
    pkl_path = tmp_path / "testfile.pkl.gz"
    test_dataset = test_helpers.load_test_dataset()
    test_dataset.get_timeseries_not_bucketed_wide_dates(CommonFields.CASES)
    test_dataset.to_compressed_pickle(pkl_path)
    assert pkl_path.stat().st_size < 900_000

    loaded_dataset = timeseries.MultiRegionDataset.from_compressed_pickle(pkl_path)

    test_helpers.assert_dataset_like(test_dataset, loaded_dataset)
