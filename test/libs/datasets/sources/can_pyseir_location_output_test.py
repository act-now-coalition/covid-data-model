import datetime
import pathlib
import pandas as pd
import pytest
from libs.datasets import can_model_output_schema as schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.enums import Intervention

pd.options.display.max_rows = 3000
pd.options.display.max_columns = 15


def _build_row(**updates):

    data = {
        "date": "2020-12-10",
        schema.ALL_HOSPITALIZED: 5,
        schema.INTERVENTION: 2,
        schema.FIPS: "36061",
        schema.BEDS: 20,
    }
    data.update(updates)
    data["date"] = pd.Timestamp(data["date"])
    return data


def _build_input_df(rows):
    return pd.DataFrame(rows)


def test_hospitalization_date():
    rows = [
        _build_row(date="2020-12-11", **{schema.ALL_HOSPITALIZED: 10}),
        _build_row(date="2020-12-12", **{schema.ALL_HOSPITALIZED: 17}),
        _build_row(date="2020-12-13", **{schema.ALL_HOSPITALIZED: 17}),
        _build_row(date="2020-12-14", **{schema.ALL_HOSPITALIZED: 8}),
    ]
    data = _build_input_df(rows)
    model_output = CANPyseirLocationOutput(data)
    # Check that it picks first date of max.
    expected_date = datetime.datetime(year=2020, month=12, day=12)
    assert model_output.peak_hospitalizations_date == expected_date


def test_shortfall():
    rows = [
        _build_row(date="2020-12-13", all_hospitalized=10, beds=11),
        _build_row(date="2020-12-14", all_hospitalized=12, beds=11),
        _build_row(date="2020-12-15", all_hospitalized=13, beds=11),
    ]
    data = _build_input_df(rows)
    model_output = CANPyseirLocationOutput(data)
    # Check that it picks first date of max.
    expected_date = datetime.datetime.fromisoformat("2020-12-14")
    assert model_output.hospitals_shortfall_date == expected_date
    assert model_output.peak_hospitalizations_shortfall == 2

    # No shortfall
    rows = [
        _build_row(date="2020-12-13", all_hospitalized=10, beds=11),
        _build_row(date="2020-12-14", all_hospitalized=12, beds=12),
    ]
    data = _build_input_df(rows)
    model_output = CANPyseirLocationOutput(data)
    # Check that it picks first date of max.
    assert not model_output.hospitals_shortfall_date


def test_load_from_path(nyc_model_output_path):

    output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)
    assert output.fips == "36061"
    assert output.intervention == Intervention.STRONG_INTERVENTION
    # manually checked values
    assert output.peak_hospitalizations_date == datetime.datetime(2020, 4, 15)
    assert not output.hospitals_shortfall_date
    assert output.latest_rt == pytest.approx(1.238822)
