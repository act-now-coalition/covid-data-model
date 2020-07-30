import logging
import re

import structlog

from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS
from libs.datasets import combined_datasets, CommonFields
from libs.datasets.combined_datasets import (
    _build_data_and_provenance,
    Override,
    _build_combined_dataset_from_sources,
    US_STATES_FILTER,
    FeatureDataSourceMap,
    _to_timeseries_rows,
)
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.sources.cmdc import CmdcDataSource
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations

from libs.datasets import JHUDataset
from libs.datasets import NYTimesDataset
from libs.datasets import CDSDataset
from libs.datasets import CovidTrackingDataSource
from libs.datasets import NevadaHospitalAssociationData

from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.timeseries import TimeseriesDataset
from test.dataset_utils_test import read_csv_and_index_fips, read_csv_and_index_fips_date, to_dict
import pandas as pd
import numpy as np
import pytest

# Tests to make sure that combined datasets are building data with unique indexes
# If this test is failing, it means that there is one of the data sources that
# is returning multiple values for a single row.


@pytest.mark.slow
def test_unique_index_values_us_timeseries():
    timeseries = combined_datasets.load_us_timeseries_dataset()
    timeseries_data = timeseries.data.set_index(timeseries.INDEX_FIELDS)
    duplicates = timeseries_data.index.duplicated()
    assert not sum(duplicates)


def test_unique_index_values_us_latest():
    latest = combined_datasets.load_us_latest_dataset()
    latest_data = latest.data.set_index(latest.INDEX_FIELDS)
    duplicates = latest_data.index.duplicated()
    assert not sum(duplicates)


# Check some counties picked arbitrarily: San Francisco/06075 and Houston (Harris County, TX)/48201
@pytest.mark.parametrize("fips", ["06075", "48201"])
def test_combined_county_has_some_data(fips):
    latest = combined_datasets.load_us_latest_dataset().get_subset(
        AggregationLevel.COUNTY, fips=fips
    )
    assert latest.data[CommonFields.POSITIVE_TESTS].all()
    assert latest.data[CommonFields.NEGATIVE_TESTS].all()
    assert latest.get_record_for_fips(fips=fips)[CommonFields.DEATHS] > 1


# Check some counties picked arbitrarily: San Francisco/06075 and Houston (Harris County, TX)/48201
@pytest.mark.parametrize("fips", ["06075", "48201"])
def test_combined_county_has_some_timeseries_data(fips):
    latest = combined_datasets.load_us_timeseries_dataset().get_subset(
        AggregationLevel.COUNTY, fips=fips
    )
    df = latest.data.set_index(CommonFields.DATE)
    assert df.loc["2020-05-01", CommonFields.CASES] > 0
    assert df.loc["2020-05-01", CommonFields.DEATHS] > 0
    if fips.startswith(
        "06"
    ):  # TODO(tom): Remove this condition when we have county data in TX too.
        assert df.loc["2020-05-01", CommonFields.POSITIVE_TESTS] > 0
        assert df.loc["2020-05-01", CommonFields.NEGATIVE_TESTS] > 0
        assert df.loc["2020-05-01", CommonFields.CURRENT_ICU] > 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_source_cls",
    [
        JHUDataset,
        CDSDataset,
        CovidTrackingDataSource,
        NevadaHospitalAssociationData,
        CmdcDataSource,
        NYTimesDataset,
        TexasHospitalizations,
    ],
)
def test_unique_timeseries(data_source_cls):
    data_source = data_source_cls.local()
    timeseries = TimeseriesDataset.build_from_data_source(data_source)
    timeseries = combined_datasets.US_STATES_FILTER.apply(timeseries)
    # Check for duplicate rows with the same INDEX_FIELDS. Sort by index so duplicates are next to
    # each other in the message if the assert fails.
    timeseries_data = timeseries.data.set_index(timeseries.INDEX_FIELDS).sort_index()
    duplicates = timeseries_data.index.duplicated(keep=False)
    assert not sum(duplicates), str(timeseries_data.loc[duplicates])


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_source_cls",
    [
        JHUDataset,
        CDSDataset,
        CovidTrackingDataSource,
        NevadaHospitalAssociationData,
        CmdcDataSource,
    ],
)
def test_expected_field_in_sources(data_source_cls):
    data_source = data_source_cls.local()
    ts = TimeseriesDataset.from_source(data_source)
    # Extract the USA data from the raw DF. Replace this with cleaner access when the DataSource makes it easy.
    rename_columns = {source: common for common, source in data_source.all_fields_map().items()}
    renamed_data = data_source.data.rename(columns=rename_columns)
    usa_data = renamed_data.loc[renamed_data["country"] == "USA"]

    assert not usa_data.empty

    states = set(usa_data["state"])

    if data_source.SOURCE_NAME == "NHA":
        assert states == {"NV"}
    else:
        good_state = set()
        for state in states:
            if re.fullmatch(r"[A-Z]{2}", state):
                good_state.add(state)
            else:
                logging.info(f"Ignoring {state} in {data_source.SOURCE_NAME}")
        assert len(good_state) >= 48


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

    results = combined_datasets._remove_padded_nans(df, ["cases"])
    expected_series = pd.Series([1, pd.NA, 2, 3], name="cases")

    pd.testing.assert_series_equal(results.cases, expected_series)


def test_build_timeseries():
    data_a = read_csv_and_index_fips_date(
        "county,state,fips,country,aggregate_level,date,cases\n"
        "Jones County,ZZ,97123,USA,county,2020-04-01,1\n"
    )
    data_b = read_csv_and_index_fips_date(
        "county,state,fips,country,aggregate_level,date,cases\n"
        "Jones County,ZZ,97123,USA,county,2020-04-01,2\n"
    )
    datasets = {"source_a": data_a, "source_b": data_b}

    combined, provenance = _build_data_and_provenance({"cases": ["source_a", "source_b"]}, datasets)
    assert combined.at[("97123", "2020-04-01"), "cases"] == 2
    assert provenance.at[("97123", "2020-04-01"), "cases"] == "source_b"

    combined, provenance = _build_data_and_provenance({"cases": ["source_b", "source_a"]}, datasets)
    assert combined.at[("97123", "2020-04-01"), "cases"] == 1
    assert provenance.at[("97123", "2020-04-01"), "cases"] == "source_a"


def test_build_latest():
    data_a = read_csv_and_index_fips(
        "county,state,fips,country,aggregate_level,date,cases\n"
        "Jones County,ZZ,97123,USA,county,2020-04-01,1\n"
        "Three County,XY,97333,USA,county,2020-04-01,3\n"
    )
    data_b = read_csv_and_index_fips(
        "county,state,fips,country,aggregate_level,date,cases\n"
        "Jones County,ZZ,97123,USA,county,2020-04-01,2\n"
    )
    datasets = {"source_a": data_a, "source_b": data_b}

    combined, provenance = _build_data_and_provenance({"cases": ["source_a", "source_b"]}, datasets)
    assert combined.at["97123", "cases"] == 2
    assert provenance.at["97123", "cases"] == "source_b"
    assert combined.at["97333", "cases"] == 3
    assert provenance.at["97333", "cases"] == "source_a"

    combined, provenance = _build_data_and_provenance({"cases": ["source_b", "source_a"]}, datasets)
    assert combined.at["97123", "cases"] == 1
    assert provenance.at["97123", "cases"] == "source_a"
    assert combined.at["97333", "cases"] == 3
    assert provenance.at["97333", "cases"] == "source_a"


def test_build_timeseries_override():
    data_a = read_csv_and_index_fips_date(
        "fips,date,m1,m2\n" "97123,2020-04-01,1,\n" "97123,2020-04-02,,\n" "97123,2020-04-03,3,3"
    )
    data_b = read_csv_and_index_fips_date(
        "fips,date,m1,m2\n" "97123,2020-04-01,,\n" "97123,2020-04-02,2,\n"
    )
    datasets = {"source_a": data_a, "source_b": data_b}

    # The combined m1 timeseries is copied from the timeseries in source_b; source_a is not used for m1
    combined, provenance = _build_data_and_provenance(
        {"m1": ["source_a", "source_b"]}, datasets, override=Override.BY_TIMESERIES
    )
    assert combined.loc["97123", "m1"].replace({np.nan: None}).tolist() == [None, 2, None]
    assert provenance.loc["97123", "m1"].replace({np.nan: None}).tolist() == [
        None,
        "source_b",
        None,
    ]

    # The combined m1 timeseries is the highest priority value for each date; source_b is higher priority for
    # both 2020-04-01 and 2020-04-02.
    combined, provenance = _build_data_and_provenance(
        {"m1": ["source_a", "source_b"]}, datasets, override=Override.BY_ROW
    )
    assert combined.loc["97123", "m1"].replace({np.nan: None}).tolist() == [None, 2, 3]
    assert provenance.loc["97123", "m1"].tolist() == ["source_b", "source_b", "source_a"]

    combined, provenance = _build_data_and_provenance(
        {"m1": ["source_b", "source_a"]}, datasets, override=Override.BY_TIMESERIES
    )
    assert combined.loc["97123", "m1"].replace({np.nan: None}).tolist() == [1, None, 3]
    assert provenance.loc["97123", "m1"].replace({np.nan: None}).tolist() == [
        "source_a",
        None,
        "source_a",
    ]

    combined, _ = _build_data_and_provenance(
        {"m1": ["source_b", "source_a"]}, datasets, override=Override.BY_ROW
    )
    assert combined.loc["97123", "m1"].replace({np.nan: None}).tolist() == [1, None, 3]


def test_build_and_and_provenance_missing_fips():
    data_a = read_csv_and_index_fips_date(
        "fips,date,m1,m2\n" "97111,2020-04-01,1,\n" "97111,2020-04-02,,\n" "97111,2020-04-03,3,3\n"
    )
    data_b = read_csv_and_index_fips_date(
        "fips,date,m1,m2\n" "97111,2020-04-01,,\n" "97111,2020-04-02,2,\n" "97444,2020-04-04,4,\n"
    )
    datasets = {"source_a": data_a, "source_b": data_b}

    # The combined m1 timeseries is copied from the timeseries in source_b; source_a is not used for m1
    combined, provenance = _build_data_and_provenance(
        {"m1": ["source_a", "source_b"], "m2": ["source_a", "source_b"]},
        datasets,
        override=Override.BY_TIMESERIES,
    )
    assert combined.loc["97444", "m1"].dropna().tolist() == [4]
    assert provenance.loc["97444", "m1"].dropna().tolist() == ["source_b"]
    assert combined.loc["97444", "m2"].dropna().tolist() == []
    assert provenance.loc["97444", "m2"].dropna().tolist() == []


@pytest.mark.slow
def test_build_combined_dataset_from_sources_smoke_test():
    # Quickly make sure _build_combined_dataset_from_sources doesn't crash when run with a small
    # number of features.

    # feature_definition = ALL_FIELDS_FEATURE_DEFINITION
    feature_definition: FeatureDataSourceMap = {
        CommonFields.CURRENT_ICU: [CmdcDataSource, CovidTrackingDataSource],
        CommonFields.CASES: [NYTimesDataset],
        CommonFields.RECOVERED: [JHUDataset],
    }

    _build_combined_dataset_from_sources(
        TimeseriesDataset, feature_definition, filter=US_STATES_FILTER,
    )

    _build_combined_dataset_from_sources(
        LatestValuesDataset, feature_definition, filter=US_STATES_FILTER,
    )


def test_melt_provenance():
    wide = read_csv_and_index_fips_date(
        "fips,date,cases,recovered\n"
        "97111,2020-04-01,source_a,source_b\n"
        "97111,2020-04-02,source_a,\n"
        "97222,2020-04-01,source_c,\n"
    )
    with structlog.testing.capture_logs() as logs:
        long = _to_timeseries_rows(wide, structlog.get_logger())

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
        long = _to_timeseries_rows(wide, structlog.get_logger())

    assert [l["event"] for l in logs] == ["Multiple rows for a timeseries"]

    assert long.to_dict() == {
        ("97111", "cases"): "source_a",
        ("97111", "recovered"): "source_b",
        ("97222", "cases"): "source_c",
    }


def test_make_latest_from_timeseries():
    data = read_csv_and_index_fips_date(
        "fips,county,state,country,date,aggregate_level,m1,m2\n"
        "97123,Smith County,ZZ,USA,2020-04-01,county,1,\n"
        "97123,Smith County,ZZ,USA,2020-04-02,county,,2\n"
    ).reset_index()
    ts = TimeseriesDataset(data)
    assert to_dict(["fips"], ts.latest_values()[["fips", "m1", "m2"]]) == {
        "97123": {"m1": 1, "m2": 2}
    }
