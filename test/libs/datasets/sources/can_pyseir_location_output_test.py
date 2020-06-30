import datetime
import pathlib
import pandas as pd
import pytest
from libs.datasets import can_model_output_schema as schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.enums import Intervention


@pytest.fixture
def nyc_model_output_path() -> pathlib.Path:
    # generated from running pyseir model output.  To update, run
    test_root = pathlib.Path(__file__).parent.parent.parent.parent
    return test_root / "data" / "pyseir" / "36061.2.json"


def _build_row(**updates):

    data = {
        "date": "2020-12-10",
        schema.ALL_HOSPITALIZED: 5,
        schema.INTERVENTION: 2,
        schema.FIPS: "36061",
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
        _build_row(date="2020-12-13", **{schema.ALL_HOSPITALIZED: 8}),
    ]
    data = _build_input_df(rows)
    model_output = CANPyseirLocationOutput(data)
    expected_date = datetime.datetime(year=2020, month=12, day=12)
    assert model_output.peak_hospitalizations_date == expected_date


def test_load_from_path(nyc_model_output_path):

    output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)
    assert output.fips == "36061"
    assert output.intervention == Intervention.OBSERVED_INTERVENTION
    assert output.peak_hospitalizations_date == datetime.datetime(2020, 4, 23)
