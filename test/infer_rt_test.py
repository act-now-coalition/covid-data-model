import pytest
import pandas as pd
import structlog
from pyseir.rt import infer_rt

from pyseir.rt.infer_rt import RtInferenceEngine
from test.mocks.inference.load_data import (
    initializeStateDataGenerator,
    DataGeneratorType,
    DataSpec,
    RateChange,
)


def test_replace_outliers_on_last_day():
    x = pd.Series([10, 10, 10, 500], [0, 1, 2, 3])

    results = infer_rt.replace_outliers(x, structlog.getLogger(), local_lookback_window=3)

    expected = pd.Series([10, 10, 10, 10], [0, 1, 2, 3])
    pd.testing.assert_series_equal(results, expected)


"""
Tests of Rt inference code using synthetically generated data for 100 days where the following are specified
1) The starting count of cases (scale)
2) One or two Rate changes - each with
    2a) time at which the change occurs (first should have t0=0)
    2b) The Rt value with which to generate growing (Rt>1) or decaying (Rt<1) values starting from t0

Note that smoothing of values smears out the transitions +/- window_size/2 days
"""

FAILURE_ERROR_FRACTION = 0.2


def test__cases_only_large_count_rising_falling_exponential():
    rt1 = 1.5
    rt2 = 0.7
    t_switch = 50

    # This configures test data generator for state NY
    initializeStateDataGenerator(
        "NY",
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=100.0,
            ratechange1=RateChange(0, rt1),
            ratechange2=RateChange(t_switch, rt2),
        ),
    )
    engine = RtInferenceEngine("36", load_data_parent="test.mocks.inference",)  # fips for NY
    df_all = engine.infer_all()

    rt = df_all["Rt_MAP_composite"]

    # Check expected values are within 10%
    assert (
        pytest.approx(rt[t_switch - 15] - 1.0, FAILURE_ERROR_FRACTION) == rt1 - 1.0
    )  # settle into first rate change
    assert (
        pytest.approx(rt[t_switch + 15] - 1.0, FAILURE_ERROR_FRACTION) == rt2 - 1.0
    )  # settle into 2nd rate change


def test__cases_only_small_count_falling_exponential():

    rt1 = 0.9
    rt2 = 0.7
    t0 = 50

    # This configures test data generator for state NY
    initializeStateDataGenerator(
        "HI",
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=50.0,
            ratechange1=RateChange(0, rt1),
            ratechange2=RateChange(t0, rt2),
        ),
    )
    engine = RtInferenceEngine("15", load_data_parent="test.mocks.inference",)  # fips for HI
    df_all = engine.infer_all()

    rt = df_all["Rt_MAP_composite"]

    # Check expected values are within 10%
    assert pytest.approx(rt[t0 - 10] - 1.0, FAILURE_ERROR_FRACTION) == rt1 - 1.0  # agree first rate
    assert pytest.approx(rt[t0 + 10] - 1.0, FAILURE_ERROR_FRACTION) == rt2 - 1.0  # agree 2nd rate


def test__cases_only_small_count_late_rising_exponential():

    rt1 = 0.95
    rt2 = 1.5
    t0 = 70

    # This configures test data generator for state NY
    initializeStateDataGenerator(
        "AK",
        DataSpec(
            generator_type=DataGeneratorType.EXP,
            disable_deaths=True,
            scale=2000.0,
            ratechange1=RateChange(0, rt1),
            ratechange2=RateChange(t0, rt2),
        ),
    )
    engine = RtInferenceEngine("02", load_data_parent="test.mocks.inference",)  # fips for AK

    tail_sup = engine.evaluate_head_tail_suppression()

    assert len(tail_sup) <= engine.window_size / 2
    assert len(tail_sup[tail_sup < 1.0]) == len(tail_sup)
    assert len(tail_sup[tail_sup > 0.0]) == len(tail_sup)
    assert tail_sup.sum() > 0.5 * len(tail_sup) and tail_sup.sum() < 0.8 * len(tail_sup)

    df_all = engine.infer_all()

    rt = df_all["Rt_MAP_composite"]

    # TODO remove adjustment here when incorporated in rt calculation
    assert (
        pytest.approx(rt[99] - 1.0, FAILURE_ERROR_FRACTION) == rt2 - 1.0
    )  # settle into 2nd specified rate at low case count
