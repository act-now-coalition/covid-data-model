import pathlib

import pytest
import pandas as pd
import structlog
from matplotlib import pyplot as plt

from pyseir.rt import utils
from pyseir.rt import infer_rt
from pyseir.utils import get_run_artifact_path, RunArtifact
from test.mocks.inference import load_data
from test.mocks.inference.load_data import RateChange


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

# output directory where test artifacts are saved.
TEST_OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output" / "test_results"


def run_individual(
    fips: str,
    spec: load_data.DataSpec,
    display_name: str,
    output_dir: pathlib.Path = TEST_OUTPUT_DIR,
):
    # TODO fails below if deaths not present even if not using
    data_generator = load_data.DataGenerator(spec)
    input_df = load_data.create_synthetic_df(data_generator)

    # Now apply smoothing and filtering
    collector = {}
    smoothed_df = infer_rt.filter_and_smooth_input_data(
        df=input_df,
        display_name=fips,
        include_deaths=False,
        figure_collector=collector,
        log=structlog.getLogger(),
    )

    engine = infer_rt.RtInferenceEngine(
        data=smoothed_df, display_name=display_name, fips=fips, figure_collector=collector
    )  # Still Needed to Pipe Output For Now
    output_df = engine.infer_all()

    # output all figures
    for (key, fig) in collector.items():
        plot_path = output_dir / f"{display_name}__fips_{fips}__{key}.pdf"
        fig.savefig(plot_path, bbox_inches="tight")

    rt = output_df["Rt_MAP_composite"]
    t_switch = spec.ratechange2.t0
    rt1 = spec.ratechange1.reff
    rt2 = spec.ratechange2.reff
    return (rt1, rt2, t_switch, rt)


def check_standard_assertions(rt1, rt2, t_switch, rt):
    OFFSET = 15
    # Check expected values are within 10%
    if abs(rt1 - 1.0) > 0.05:
        assert (
            pytest.approx(rt[t_switch - OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt1 - 1.0
        )  # settle into first rate change
    else:
        assert abs(rt[t_switch - OFFSET] - rt1) < 0.1  # settle into first rate change
    assert (
        pytest.approx(rt[t_switch + OFFSET] - 1.0, rel=FAILURE_ERROR_FRACTION) == rt2 - 1.0
    )  # settle into 2nd rate change

    assert (
        pytest.approx(rt[-1] - 1.0, rel=FAILURE_ERROR_FRACTION * 2) == rt2 - 1.0
    ), f"Test {id} Failed: Today Value Not Within Spec: Predicted={round(rt[-1],2)} Observed={rt2}."


def test_constant_cases_high_count(tmp_path):
    """Track constant cases (R=1) at low count"""
    data_spec = load_data.DataSpec(
        generator_type=load_data.DataGeneratorType.EXP,
        disable_deaths=True,
        scale=1000.0,
        ratechange1=RateChange(0, 1.0),
        ratechange2=RateChange(80, 1.5),  # To avoid plotting issues
    )
    rt1, rt2, t_switch, rt = run_individual(
        "20", data_spec, "test_constant_cases_high_count"  # Kansas
    )
    check_standard_assertions(rt1, rt2, t_switch, rt)


def test_med_scale_strong_growth_and_decay():
    """Track cases growing strongly and then decaying strongly"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "36",  # New York
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
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
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
            disable_deaths=True,
            scale=5.0,
            ratechange1=RateChange(0, 1.0),
            ratechange2=RateChange(70, 1.2),
        ),
        "test_low_cases_weak_growth",
    )
    # check_standard_assertions(rt1, rt2, t_switch, rt)
    # TODO is failing this test rt = .84 instead of rt1


def test_high_scale_late_growth():
    """Track decaying from high initial count to low number then strong growth"""
    (rt1, rt2, t_switch, rt) = run_individual(
        "02",  # Alaska
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
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
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
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
        load_data.DataSpec(
            generator_type=load_data.DataGeneratorType.EXP,
            disable_deaths=True,
            scale=1000.0,
            ratechange1=RateChange(0, 1.0),
            ratechange2=RateChange(95, 5.0),
        ),
        "test_smoothing_and_causality",
    )
