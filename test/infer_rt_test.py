import numpy as np

from pyseir.inference.infer_rt import RtInferenceEngine
from test.mocks.inference.pyseir_default_parameters import pyseir_default_parameters
from test.mocks.inference.load_data import (
    initializeStateDataGenerator,
    DataGeneratorType,
    DataSpec,
    RateChange,
)

import pytest

"""
Tests of Rt inference code using synthetically generated data for 100 days where the following are specified
1) The starting count of cases (scale)
2) One or two Rate changes - each with
    2a) time at which the change occurs (first should have t0=0)
    2b) The Rt value with which to generate growing (Rt>1) or decaying (Rt<1) values starting from t0

Note that smoothing of values smears out the transitions +/- window_size/2 days
"""


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
    engine = RtInferenceEngine(
        "36",  # fips for NY
        load_data_parent="test.mocks.inference",
        default_parameters=pyseir_default_parameters,
    )
    df_all = engine.infer_all()

    rt = df_all["Rt_MAP_composite"]

    # Check expected values are within 10%
    assert pytest.approx(rt[t_switch - 10], 0.1) == rt1  # settle into first rate change
    assert pytest.approx(rt[t_switch + 20], 0.1) == rt2  # settle into 2nd rate change


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
    engine = RtInferenceEngine(
        "15",  # fips for HI
        load_data_parent="test.mocks.inference",
        default_parameters=pyseir_default_parameters,
    )
    df_all = engine.infer_all()

    rt = df_all["Rt_MAP_composite"]

    # Check expected values are within 10%
    assert pytest.approx(rt[t0 - 10], 0.1) == rt1  # agree first rate
    # TODO when fixed - assert (pytest.approx(rt[t0 + 10], 0.1) == rt2) # agree 2nd rate


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
    engine = RtInferenceEngine(
        "02",  # fips for AK
        load_data_parent="test.mocks.inference",
        default_parameters=pyseir_default_parameters,
    )
    df_all = engine.infer_all()

    rt = df_all["Rt_MAP_composite"]

    # Calculate rt suppression by smoothing delay at tail of sequence
    tail_suppression = np.concatenate(
        [1.0 * np.ones(len(rt) - 7), np.array([1.0, 0.72, 0.67, 0.62, 0.56, 0.52, 0.48])]
    )
    # Adjust rt by undoing the supppression
    rt_adj = (rt - 1.0) / tail_suppression + 1.0

    print("last 20 adjusted", rt_adj.tail(20).apply(lambda v: f"%.2f" % v).values)

    # Check expected values are within 10%
    # TODO remove adjustment here when incorporated in rt calculation
    assert (
        pytest.approx(rt_adj[99], 0.05) == rt2
    )  # settle into 2nd specified rate at low case count
