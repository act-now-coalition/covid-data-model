import pytest
import pandas as pd
import structlog
from pyseir.rt import utils
import pyseir.rt.infer_rt as infer_rt
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

synthetic_tests = [
    dict(
        id="01",
        spec=DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=100.0,
            ratechange1=RateChange(0, 1.5),
            ratechange2=RateChange(50, 0.7),
        ),
    ),
    dict(
        id="02",
        spec=DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=2000.0,
            ratechange1=RateChange(0, 0.95),
            ratechange2=RateChange(70, 1.5),
        ),
    ),
    dict(
        id="06",
        spec=DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=50.0,
            ratechange1=RateChange(0, 0.9),
            ratechange2=RateChange(50, 0.7),
        ),
    ),
]


def run_individual(test):
    input_df = create_synthetic_df(DataGenerator(test["spec"]))
    engine = infer_rt.RtInferenceEngine(
        data=input_df, display_name=f"Test {test['id']}", fips=test["id"]
    )  # Still Needed to Pipe Output For Now
    output_df = engine.infer_all()

    rt = output_df["Rt_MAP_composite"]
    OFFSET = 15
    t_switch = test["spec"].ratechange2.t0
    rt1 = test["spec"].ratechange1.reff
    rt2 = test["spec"].ratechange2.reff

    # Check expected values are within 10%
    assert (
        pytest.approx(rt[t_switch - OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt1 - 1.0
    ), "First Stage Failed to Converge Within Tolerance"  # settle into first rate change
    assert (
        pytest.approx(rt[t_switch + OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt2 - 1.0
    ), "Second Stage Failed to Converge Within Tolerance"  # settle into 2nd rate change


def test_all():
    for test_spec in synthetic_tests:
        run_individual(test_spec)


# TODO: Alex -> Add a tail specific test
