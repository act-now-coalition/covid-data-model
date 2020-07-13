import pytest
import pandas as pd
import structlog

from pyseir.rt import utils
import pyseir.rt.infer_rt as infer_rt
from pyseir.utils import get_run_artifact_path, RunArtifact
from test.mocks.inference.load_data import (
    DataGeneratorType,
    DataSpec,
    RateChange,
    DataGenerator,
    create_synthetic_df,
)


def test_replace_outliers_on_last_day():
    x = pd.Series([10, 10, 10, 500], [0, 1, 2, 3])

    results = utils.replace_outliers(x, structlog.getLogger(), local_lookback_window=3)

    expected = pd.Series([10, 10, 10, 10], [0, 1, 2, 3])
    pd.testing.assert_series_equal(results, expected)


"""
Tests of Rt inference code using synthetically generated data for 100 days where the following are
specified:
1) The starting count of cases (scale)
2) One or two Rate changes - each with
    2a) time at which the change occurs (first should have t0=0)
    2b) The Rt value with which to generate growing (Rt>1) or decaying (Rt<1) values

Note that smoothing of values smears out the transitions +/- window_size/2 days
"""

FAILURE_ERROR_FRACTION = 0.2

""" test_specs = {
    "36": DataSpec(
        generator_type=DataGeneratorType.EXP,
        disable_deaths=True,
        scale=100.0,
        ratechange1=RateChange(0, 1.5),
        ratechange2=RateChange(50, 0.7),
    ),  # New York
    "02": DataSpec(
        generator_type=DataGeneratorType.EXP,
        disable_deaths=True,
        scale=2000.0,
        ratechange1=RateChange(0, 0.95),
        ratechange2=RateChange(70, 1.5),
    ),  # Alaska
    "06": DataSpec(
        generator_type=DataGeneratorType.EXP,
        disable_deaths=True,
        scale=50.0,
        ratechange1=RateChange(0, 0.9),
        ratechange2=RateChange(50, 0.7),
    ),  # California
    "56": DataSpec(
        generator_type=DataGeneratorType.EXP,
        disable_deaths=True,
        scale=1000.0,
        ratechange1=RateChange(0, 1.0),
        ratechange2=RateChange(95, 5.0),
    ),  # Wyoming
    "50": DataSpec(
        generator_type=DataGeneratorType.EXP,
        disable_deaths=True,
        scale=5.0,
        ratechange1=RateChange(0, 1.0),
        ratechange2=RateChange(60, 1.2),
    ),  # Vermont
} """


def run_individual(id, spec, display_name):
    # TODO fails below if deaths not present even if not using
    input_df = create_synthetic_df(DataGenerator(spec))

    # Now apply smoothing and filtering
    plot_path = get_run_artifact_path(id + " " + display_name, RunArtifact.RT_SMOOTHING_REPORT)
    smoothed_df = infer_rt.filter_and_smooth_input_data(
        df=input_df, include_deaths=False, plot_path=plot_path
    )

    engine = infer_rt.RtInferenceEngine(
        data=smoothed_df, display_name=display_name, fips=id
    )  # Still Needed to Pipe Output For Now
    output_df = engine.infer_all()

    rt = output_df["Rt_MAP_composite"]
    t_switch = spec.ratechange2.t0
    rt1 = spec.ratechange1.reff
    rt2 = spec.ratechange2.reff
    return (rt1, rt2, t_switch, rt)


def check_standard_assertions(rt1, rt2, t_switch, rt):
    OFFSET = 15
    # Check expected values are within 10%
    assert (
        pytest.approx(rt[t_switch - OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt1 - 1.0
    ), "First Stage Failed to Converge Within Tolerance"  # settle into first rate change
    assert (
        pytest.approx(rt[t_switch + OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt2 - 1.0
    ), "Second Stage Failed to Converge Within Tolerance"  # settle into 2nd rate change

    assert (
        pytest.approx(rt[-1] - 1.0, rel=FAILURE_ERROR_FRACTION * 2) == rt2 - 1.0
    ), f"Test {id} Failed: Today Value Not Within Spec: Predicted={round(rt[-1],2)} Observed={rt2}."


def test_constant_cases_high_count():
    """Track constant cases (R=1) at low count"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "20",  # Kansas
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=2.0,
            ratechange1=RateChange(0, 1.0),
            ratechange2=RateChange(80, 1.5),  # To avoid plotting issues
        ),
        "test_constant_cases_high_count",
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


def test_med_scale_strong_growth_and_decay():
    """Track cases growing strongly and then decaying strongly"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "36",  # New York
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=100.0,
            ratechange1=RateChange(0, 1.5),
            ratechange2=RateChange(50, 0.7),
        ),
        "test_med_scale_strong_growth_and_decay",
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


def test_low_cases_weak_growth():
    """Track with low scale (count = 5) and slow growth"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "50",  # Vermont
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=5.0,
            ratechange1=RateChange(0, 1.0),
            ratechange2=RateChange(70, 1.2),
        ),
        "test_low_cases_weak_growth",
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


def test_high_scale_late_growth():
    """Track decaying from high initial count to low number then strong growth"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "02",  # Alaska
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=2000.0,
            ratechange1=RateChange(0, 0.95),
            ratechange2=RateChange(70, 1.5),
        ),
        "test_high_scale_late_growth",
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


def test_low_scale_two_decays():
    """Track low scale decay at two different rates"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "06",  # California
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=50.0,
            ratechange1=RateChange(0, 0.9),
            ratechange2=RateChange(50, 0.7),
        ),
        "test_low_scale_two_decays",
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


def test_smoothing_and_causality():
    run_individual(
        "56",  # Wyoming
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=1000.0,
            ratechange1=RateChange(0, 1.0),
            ratechange2=RateChange(95, 5.0),
        ),
        "test_smoothing_and_causality",
    )


# TODO: Alex -> Add a tail specific test
