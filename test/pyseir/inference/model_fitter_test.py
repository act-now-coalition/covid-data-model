import dataclasses
from typing import Optional

import pytest

from libs import pipeline
from pyseir.inference import model_fitter
import pandas as pd

# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error", "ignore::libs.pipeline.BadFipsWarning")


def test_get_pyseir_fitter_initial_conditions():
    params = [
        "R0",
        "t0",
        "eps",
        "t_break",
        "eps2",
        "t_delta_phases",
        "test_fraction",
        "hosp_fraction",
    ]

    region = pipeline.Region.from_fips("01")
    mapping = model_fitter.RegionalInput.from_state_region(
        region
    ).get_pyseir_fitter_initial_conditions(params)

    assert mapping["R0"] > 0
    assert mapping["t0"] > 0


def test_get_pyseir_fitter_initial_conditions_none():
    params = [
        "R0",
        "t0",
        "eps",
        "t_break",
        "eps2",
        "t_delta_phases",
        "test_fraction",
        "hosp_fraction",
    ]
    # A mock of model_fitter.ModelFitter
    MockModelFitter = dataclasses.make_dataclass(
        "MockModelFitter",
        [("fit_results", Optional[pd.DataFrame], dataclasses.field(default=None))],
    )
    mock_state_fitter = MockModelFitter()

    # Loving County, Texas (population 169) does not have initial conditions in our data file
    region = pipeline.Region.from_fips("48301")
    mapping = model_fitter.RegionalInput.from_substate_region(
        region, state_fitter=mock_state_fitter,
    ).get_pyseir_fitter_initial_conditions(params)

    assert mapping == {}
