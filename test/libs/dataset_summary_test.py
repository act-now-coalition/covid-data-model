import pandas as pd
import numpy as np
import pytest

import libs.qa.dataset_summary_gen
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.sources.covid_county_data import CovidCountyDataDataSource
from libs.datasets.timeseries import TimeseriesDataset
from libs.qa.dataset_summary import summarize_timeseries_fields


def test_generate_field_multiple_data_points():

    values = [None, None, 10, 30, 32, None, 40, None]
    series = pd.Series(values, index=[f"2020-07-0{i + 1}" for i in range(len(values))])

    results = libs.qa.dataset_summary_gen.generate_field_summary(series)
    expected = {
        "has_value": True,
        "largest_delta": 20.0,
        "largest_delta_date": "2020-07-04",
        "latest_value": 40.0,
        "max_date": "2020-07-07",
        "max_value": 40.0,
        "min_date": "2020-07-03",
        "min_value": 10.0,
        "num_observations": 4,
    }
    assert expected == results.to_dict()


def test_generate_field_one_data_points():

    values = [None, None, 10, None, None, None, None, None]
    series = pd.Series(values, index=[f"2020-07-0{i + 1}" for i in range(len(values))])

    results = libs.qa.dataset_summary_gen.generate_field_summary(series)
    expected = {
        "has_value": True,
        "largest_delta": np.nan,
        "largest_delta_date": None,
        "latest_value": 10.0,
        "max_date": "2020-07-03",
        "max_value": 10.0,
        "min_date": "2020-07-03",
        "min_value": 10.0,
        "num_observations": 1,
    }
    assert expected == results.to_dict()


def test_generate_field_no_data_points():

    variable_name = "cases"
    values = [None, None, None, None, None, None, None, None]
    series = pd.Series(values, index=[f"2020-07-0{i + 1}" for i in range(len(values))])

    results = libs.qa.dataset_summary_gen.generate_field_summary(series)
    expected = {
        "has_value": False,
        "largest_delta": None,
        "largest_delta_date": None,
        "latest_value": None,
        "max_date": None,
        "max_value": None,
        "min_date": None,
        "min_value": None,
        "num_observations": 0,
    }
    assert expected == results.to_dict()


@pytest.mark.slow
def test_summarize_timeseries_fields_with_some_real_data():
    data_source = CovidCountyDataDataSource.local()
    ts = TimeseriesDataset.from_source(data_source)
    summary = summarize_timeseries_fields(
        ts.data.loc[lambda df: df[CommonFields.FIPS].str.startswith("06")]
    )
    assert not summary.empty
    cases_summary = summary.loc[("06025", "cases"), :]
    assert summary.loc[("06025", "cases"), "max_value"] > 7000
    assert summary.loc[("06025", "cases"), "max_date"] > pd.to_datetime("2020-08-01")
    assert summary.loc[("06025", "cases"), "largest_delta_date"] > pd.to_datetime("2020-04-01")
    assert cases_summary["has_value"] == True
    assert cases_summary["num_observations"] > 100
