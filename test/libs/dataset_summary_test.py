import pandas as pd
import numpy as np
from libs.qa import dataset_summary


def test_generate_field_multiple_data_points():

    variable_name = "cases"
    values = [None, None, 10, 30, 32, None, 40, None]
    series = pd.Series(
        values, index=[(variable_name, f"2020-07-0{i + 1}") for i in range(len(values))]
    )

    results = dataset_summary.generate_field_summary(series)
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

    variable_name = "cases"
    values = [None, None, 10, None, None, None, None, None]
    series = pd.Series(
        values, index=[(variable_name, f"2020-07-0{i + 1}") for i in range(len(values))]
    )

    results = dataset_summary.generate_field_summary(series)
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
    series = pd.Series(
        values, index=[(variable_name, f"2020-07-0{i + 1}") for i in range(len(values))]
    )

    results = dataset_summary.generate_field_summary(series)
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
