from libs import pipeline
from pyseir.inference import model_fitter


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

    region = pipeline.Region.from_fips("99")
    mapping = model_fitter.RegionalInput.from_state_region(
        region
    ).get_pyseir_fitter_initial_conditions(params)

    assert mapping == {}
