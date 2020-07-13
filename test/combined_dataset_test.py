import logging
import pathlib
import re

import pytest
import structlog
from more_itertools import one

from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS
from libs.datasets import combined_datasets, CommonFields
from libs.datasets.combined_datasets import build_timeseries, _build_dataframe
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.sources.cmdc import CmdcDataSource
from libs.datasets.sources.texas_hospitalizations import TexasHospitalizations

from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import JHUDataset
from libs.datasets import NYTimesDataset
from libs.datasets import CDSDataset
from libs.datasets import CovidTrackingDataSource
from libs.datasets import NevadaHospitalAssociationData
from covidactnow.datapublic.common_df import write_df_as_csv, read_csv_to_indexed_df
from datetime import datetime

from io import StringIO

from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.sources import cds_dataset
from libs.datasets.timeseries import TimeseriesDataset
from test.dataset_utils_test import to_dict
import pandas as pd
import numpy as np
import pytest

# Tests to make sure that combined datasets are building data with unique indexes
# If this test is failing, it means that there is one of the data sources that
# is returning multiple values for a single row.
from test.dataset_utils_test import to_dict


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




def read_csv_str(csv_str: str) -> pd.DataFrame:
    return pd.read_csv(
        StringIO(csv_str),
        parse_dates=[CommonFields.DATE],
        dtype={CommonFields.FIPS: str},
        low_memory=False,
    )


def test_build_timeseries():
    data_a = read_csv_str(
        "county,state,fips,country,aggregate_level,date,cases\n"
        "Jones County,ZZ,97123,USA,county,2020-04-01,1\n"
    ).set_index(COMMON_FIELDS_TIMESERIES_KEYS)
    data_b = read_csv_str(
        "county,state,fips,country,aggregate_level,date,cases\n"
        "Jones County,ZZ,97123,USA,county,2020-04-01,2\n"
    ).set_index(COMMON_FIELDS_TIMESERIES_KEYS)
    datasets = {"source_a": data_a, "source_b": data_b}

    combined = _build_dataframe({"cases": ["source_a", "source_b"]}, datasets)
    assert combined.at[("97123", "2020-04-01"), "cases"] == 2

    combined = _build_dataframe({"cases": ["source_b", "source_a"]}, datasets)
    assert combined.at[("97123", "2020-04-01"), "cases"] == 1


def test_build_latest():
    data_a = (
        read_csv_str(
            "county,state,fips,country,aggregate_level,date,cases\n"
            "Jones County,ZZ,97123,USA,county,2020-04-01,1\n"
            "Three County,XY,97333,USA,county,2020-04-01,3\n"
        )
        .groupby(CommonFields.FIPS)
        .last()
    )
    data_b = (
        read_csv_str(
            "county,state,fips,country,aggregate_level,date,cases\n"
            "Jones County,ZZ,97123,USA,county,2020-04-01,2\n"
        )
        .groupby(CommonFields.FIPS)
        .last()
    )
    datasets = {"source_a": data_a, "source_b": data_b}

    combined = _build_dataframe({"cases": ["source_a", "source_b"]}, datasets)
    assert combined.at["97123", "cases"] == 2
    assert combined.at["97333", "cases"] == 3

    combined = _build_dataframe({"cases": ["source_b", "source_a"]}, datasets)
    assert combined.at["97123", "cases"] == 1
    assert combined.at["97333", "cases"] == 3


def test_build_timeseries_override():
    data_a = read_csv_str(
        "fips,date,cases\n" "97123,2020-04-01,1\n" "97123,2020-04-02,\n"
    ).set_index(COMMON_FIELDS_TIMESERIES_KEYS)
    data_b = read_csv_str(
        "fips,date,cases\n" "97123,2020-04-01,\n" "97123,2020-04-02,2\n"
    ).set_index(COMMON_FIELDS_TIMESERIES_KEYS)
    datasets = {"source_a": data_a, "source_b": data_b}

    combined = _build_dataframe({"cases": ["source_a", "source_b"]}, datasets)
    assert combined.loc["97123", "cases"].replace({np.nan: None}).tolist() == [None, 2]

    combined = _build_dataframe({"cases": ["source_b", "source_a"]}, datasets)
    assert combined.loc["97123", "cases"].replace({np.nan: None}).tolist() == [1, None]
