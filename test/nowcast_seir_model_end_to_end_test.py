import pathlib

import pytest
import pandas as pd
import numpy as np
import math
from random import choices, randrange
import structlog
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, time

from pyseir.models.seir_model import (
    Demographics,
    ramp_function,
    SEIRModel,
    steady_state_ratios,
    NowcastingSEIRModel,
    ModelRun,
)
from pyseir.rt.constants import InferRtConstants
from pyseir.models.historical_data import HistoricalData, adjust_rt_to_match_cases

# rom pyseir.utils import get_run_artifact_path, RunArtifact
from test.mocks.inference import load_data
from test.mocks.inference.load_data import RateChange

TEST_OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output" / "test_results"


def test_run_new_model_incrementally():
    """
    Demonstrates how to minimally run the new model incrementally from an initial set of
    observables (new cases, hospitalizations, new deaths) and with given assumptions (for
    r(t), test_rate and median_age) into the future.
    """
    # These are values for FL on
    nC_initial = 800.0
    nD_initial = 30.0
    H_initial = 750.0  # Guess as there is no data

    t_list = np.linspace(147, 215, 215 - 147 + 1)
    (rt, nc, tests, _1, _2) = HistoricalData.get_state_data_for_dates(
        "FL", t_list, as_functions=True
    )

    start = datetime.now()
    # TODO pass in history and apply constraint at last time period (today)
    run = ModelRun(
        NowcastingSEIRModel(),
        20e6,  # N
        t_list,
        tests,
        rt,
        case_median_age_f=ramp_function(t_list, 48, 37),
        initial_compartments={"nC": nC_initial, "H": H_initial, "nD": nD_initial},
        auto_initialize_other_compartments=True,
        auto_calibrate=True,
    )
    (results, ratios, fig) = run.execute_dataframe_ratios_fig()

    # Should finish in less that 1 second
    elapsed = (datetime.now() - start).seconds
    assert elapsed < 1

    fig.savefig(TEST_OUTPUT_DIR / "test_run_new_model_incrementally.pdf", bbox_inches="tight")


def test_historical_peaks_positivity_to_real_cfr():
    peaks = pd.read_csv("test/data/historical/historical_peaks.csv")
    early_peaks = peaks[peaks["when"] == "Apr-May"]
    late_peaks = peaks[peaks["when"] == "Jun-Jul"]

    early_peaks["adjusted"] = early_peaks["ratio_cases_to_deaths"] / 0.36

    fig = plt.figure(facecolor="w", figsize=(10, 6))
    plt.scatter(
        late_peaks["peak_positivity_percent"],
        1.0 / late_peaks["ratio_cases_to_deaths"],
        color="g",
        label="late peaks",
    )
    plt.scatter(
        early_peaks["peak_positivity_percent"],
        1.0 / early_peaks["adjusted"],
        color="r",
        label="early peaks (* .36)",
    )

    plt.plot([0, 40], [0.01, 0.032])

    plt.ylim((0, 0.05))
    plt.ylabel("Peak deaths to cases (CFR)")
    plt.xlabel("Max test positivity (%)")
    plt.legend()
    fig.savefig(TEST_OUTPUT_DIR / "test_historical_peaks_positivity_to_real_cfr.pdf")

    assert True


def test_random_periods_interesting_states():
    states = HistoricalData.get_states()
    early_start = {"NY": 75, "NJ": 80, "CT": 80}

    # Good if not listed (about 35)
    # ok-ish: AL, CO, DC, GA, IL, NJ, NM, OK, RI,
    # bad: CA, HI, LA, SD, VT
    # weird, something wrong: WY
    for state in states:
        num_days = 60
        earliest = early_start[state] if state in early_start else 90
        start = randrange(
            earliest, 218 - num_days, 1
        )  # Hospitalizations not available before mid Apr
        t_list = np.linspace(start, start + num_days, num_days + 1)
        t0 = t_list[0]
        (rt, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates(
            state, t_list, as_functions=False
        )
        (average_R, growth_ratio, adj_r_f) = adjust_rt_to_match_cases(
            lambda t: rt[t], lambda t: nc[t], t_list
        )

        run = ModelRun(
            NowcastingSEIRModel(),
            20e6,  # N
            t_list,
            lambda t: tests[t],
            adj_r_f,
            case_median_age_f=Demographics.median_age_f(),
            # initial_compartments={"nC": nc(t0), "H": h(t0), "nD": nd(t0)},
            historical_compartments={"nC": nc, "H": h, "nD": nd},
            auto_initialize_other_compartments=True,
            auto_calibrate=True,
        )
        (results, ratios, fig) = run.execute_dataframe_ratios_fig()
        fig.savefig(
            TEST_OUTPUT_DIR / (f"test_random_periods_%s_%d.pdf" % (state, start)),
            bbox_inches="tight",
        )


def test_reproduce_FL_late_peak():
    """
    TODO fix to use historical data
    Reproduce behaviour of late peak in Florida where (as of ~Aug 1)
    - was 800 cases per day (20k tests per day with 4% positivity) steady for some time with 30 deaths per day
    - then R(t) went to 1.1 and 1.3 and 1.0 each for two weeks
    - deaths didn't rise (by 20%) until at least 4 weeks later -31-(-59)=28 than cases did
    - cases peaked at 120000/day and deaths at 185/day (2 weeks later)
    - with median age of recent cases = 37 years old
    """
    # new cases per day
    nC_initial = 800.0
    nC_max_exp = 12000.0
    nC_final_exp = 9600.0
    nD_initial = 30.0
    H_initial = 750.0  # Guess as there is no data

    nD_max_exp = 185.0  # new deaths per day
    death_peak_delay_exp = 14  # days
    death_rising_start_delay = 28  # days
    # TODO change days below to be relative to Jan 1 not Aug 1

    # Setup model and times
    model = NowcastingSEIRModel()
    t_list = np.linspace(-63, 0, 64)
    rt = FL_rt_divoc_times()

    # TODO double check this with exact R(t) we have for FL as seemed to have to adjust up
    up_R = [rt(t) for t in t_list[:-14]]
    average_R = sum(up_R) / len(up_R)
    growth_ratio = (nC_max_exp / nC_initial) / math.exp(
        (average_R - 1) * (len(t_list) - 15) / model.serial_period
    )
    assert growth_ratio > 0.95 and growth_ratio < 1.05

    # Initialize model run and execute it
    run = ModelRun(
        model,
        20e6,
        t_list,
        FL_test_rate_divoc_times(),
        rt,
        case_median_age_f=ramp_function(t_list, 48, 37),
        initial_compartments={"nC": nC_initial},
        auto_initialize_other_compartments=True,
        auto_calibrate=True,
    )
    (results, ratios, fig) = run.execute_dataframe_ratios_fig()

    # TODO better check that people are conserved (see_model.py ModelRun._time_step)
    # especially now that deaths not tracked
    rN = results.tail(1).to_dict(orient="records")[0]
    r0 = results.head(1).to_dict(orient="records")[0]
    start = r0["S"] + r0["E"] + r0["I"] + r0["A"] + r0["C"] + r0["H"] + r0["nD"] + r0["R"]
    end = rN["S"] + rN["E"] + rN["I"] + rN["A"] + rN["C"] + rN["H"] + rN["nD"] + rN["R"]
    # assert abs(start - end) < 10000.0

    # Check that new cases as expected
    assert rN["nC"] / nC_final_exp > 0.95 and rN["nC"] / nC_final_exp < 1.05

    # Check that daily deaths as expected - barely passing now
    # TODO take deaths at the peak not the end
    ratio = rN["nD"] / nD_max_exp
    assert ratio > 0.9 and ratio < 1.1

    # Check that peak in deaths happens at the right time
    observed_peak_delay = results["nD"].argmax() - results["nC"].argmax()
    diff_peak_delay = abs(observed_peak_delay - death_peak_delay_exp)
    assert diff_peak_delay < 5

    # Check that deaths start to ramp up (by 20%) as late as observed compared to cases
    start_deaths = r0["nD"]
    start_cases = r0["nC"]
    cases_at = results[results["nC"] > 1.2 * start_cases].head(1).index.values[0]
    deaths_at = results[results["nD"] > 1.2 * start_deaths].head(1).index.values[0]
    diff_start_delay = abs(death_rising_start_delay - (deaths_at - cases_at))
    assert diff_start_delay < 5  # TODO less than 5 fails

    # TODO check if deaths and hospitalizations are exactly right (slopes close) at the start of the run


# TODO add a test for a recent peak with reasonably low test positivity
