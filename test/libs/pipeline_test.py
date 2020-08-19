from libs import pipeline


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

    mapping = pipeline.Region.from_fips("01").get_pyseir_fitter_initial_conditions(params)

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

    mapping = pipeline.Region.from_fips("99").get_pyseir_fitter_initial_conditions(params)

    assert mapping == {}
