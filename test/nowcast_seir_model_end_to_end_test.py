import pathlib

import pytest
import pandas as pd
import numpy as np
import math
from numbers import Number
from scipy import stats
from random import choices, randrange
import structlog
from matplotlib import pyplot as plt
from datetime import datetime, timedelta, time

from pyseir.models.nowcast_seir_model import (
    Demographics,
    ramp_function,
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

    # Time period for the run (again relative to Jan 1, 2020)
    t_list = np.linspace(
        147, 230, 230 - 147 + 1
    )  # Here starts from when FL started to ramp up cases

    # Need assumptions for R(t) and testing_rate(t) for the future as inputs
    (rt, ignore_0, tests, ignore_1, ignore_2) = HistoricalData.get_state_data_for_dates(
        "FL", t_list, as_functions=True
    )  # Here taken from historical data but typically will use R(t) projections

    start = datetime.now()
    # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
    run = ModelRun(
        NowcastingSEIRModel(
            delay_ci_h=5
        ),  # creating a default model here with no trained parameters
        20e6,  # N, population of jurisdiction
        t_list,
        tests,
        rt,
        case_median_age_f=Demographics.median_age_f("FL"),
        # ramp_function( t_list, 48, 37),  # Optional, will use US defaults if not supplied
        initial_compartments={"nC": 800.0, "H": 750.0, "nD": 30.0},  # S and I can also be supplied
        auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
        auto_calibrate=True,  # Override some model constants to ensure continuity with initial compartments
    )

    # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
    (results, ratios, fig) = run.execute_dataframe_ratios_fig()

    # Should have finished in less that 1 second
    elapsed = (datetime.now() - start).seconds

    fig.savefig(TEST_OUTPUT_DIR / "test_run_new_model_incrementally.pdf", bbox_inches="tight")

    assert elapsed < 1


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


def test_ratio_evolution():
    t_list = np.linspace(0, 230, 230 + 1)
    sets = {
        "early": {"states": ["NY", "NJ", "CT"], "delay": 0},
        "late": {"states": ["FL", "TX", "GA", "CA", "AZ"], "delay": 14},
        "steady": {"states": ["PA", "IL"], "delay": 7},
    }

    model = NowcastingSEIRModel()
    for name, s in sets.items():
        fig, ax = plt.subplots()
        for state in s["states"]:
            (rt, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates(
                state, t_list, as_functions=False
            )
            fh = h / (nc * model.t_i)  # .shift(7)
            fd = (nd * model.t_h) / h  # .shift(7)
            pos = nc / tests
            hdelay = h.shift(s["delay"])

            plt.scatter(hdelay.values, nd.values, marker=".", label=state)
            plt.plot([hdelay.values[-1]], [nd.values[-1]], marker="o")

        plt.yscale("log")
        plt.xscale("log")
        # plt.ylim((0.01, 1.0))
        fig.legend()
        fig.savefig(TEST_OUTPUT_DIR / (f"test_ratio_evolution_%s_states.pdf" % name))


def test_random_periods_interesting_states():
    states = HistoricalData.get_states()
    starts = {"early": 90, "recent": 170}
    early_start = {
        "NY": 75,
        "NJ": 80,
        "CT": 80,
        "WY": 100,
        "AK": 115,
        "AZ": 100,
        "DC": 100,
        "CA": 115,
        "GA": 120,
        "IL": 95,
        "IN": 110,
        "KY": 95,
        "MD": 106,
        "ME": 98,
        "MI": 95,
        "MS": 105,
        "MT": 95,
        "NH": 100,
        "NV": 95,
        "OH": 120,
        "PA": 95,
        "SD": 110,
        "UT": 123,
        "VA": 95,
        "WI": 97,
        "WV": 95,
    }

    calibrations = []

    for state in states:  # ["TX"]:  #
        # Do a test early and late in the pandemic for each state
        for (when, std_start) in starts.items():
            num_days = 60  # duration of test
            earliest = (
                early_start[state] if (when == "early" and state in early_start) else std_start
            )
            start = earliest
            # randrange(
            #     earliest, min(218 - num_days, 30 + earliest), 1
            # )  # Pick a random start time within a range
            t_list = np.linspace(start, start + num_days, num_days + 1)
            t0 = t_list[0]
            h_delay = 0.0 if when == "early" else 7.0

            # Get historical data for that state and adjust R(t) for long term bias
            (rt, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates(
                state, t_list, as_functions=False
            )
            (average_R, growth_ratio, adj_r_f) = adjust_rt_to_match_cases(
                lambda t: rt[t], lambda t: nc[t], t_list
            )

            # Run the model with history for initial constraints and to add actuals to charts
            run = ModelRun(
                NowcastingSEIRModel(delay_ci_h=h_delay, death_delay=0),
                20e6,  # N
                t_list,
                lambda t: tests[t],
                adj_r_f,
                case_median_age_f=Demographics.median_age_f(state),
                # initial_compartments={"nC": nc(t0), "H": h(t0), "nD": nd(t0)},
                historical_compartments={"nC": nc, "H": h, "nD": nd},
                auto_initialize_other_compartments=True,
                auto_calibrate=True,
            )
            (results, ratios, fig) = run.execute_dataframe_ratios_fig()
            calibrations.append(
                (state, when, run.model.lr_fh[0] * run.model.lr_fd[0], run.model.lr_fh[0])
            )

            fig.savefig(
                TEST_OUTPUT_DIR
                / (
                    f"test_random_periods_%s_%s_start=%d_calibration=(%.2f,%.2f->%.2f).pdf"
                    % (
                        state,
                        when,
                        start,
                        run.model.lr_fh[0],
                        run.model.lr_fd[0],
                        run.model.lr_fh[0] * run.model.lr_fd[0],
                    )
                ),
                bbox_inches="tight",
            )
    df = pd.DataFrame(calibrations, columns=["state", "when", "fh0", "fd0"])
    for when in ["early", "recent"]:
        sub = df[df["when"] == when]
        fig, ax = plt.subplots()
        ax.scatter(sub.fh0, sub.fd0)
        for i in sub.index:
            ax.annotate(sub["state"][i], (sub["fh0"][i], sub["fd0"][i]))

        plt.xlabel("fd0*fh0 (gmean =%.2f)" % stats.gmean(sub.fh0 * sub.fd0))
        plt.ylabel("fd0/fh0 (gmean =%.2f)" % stats.gmean(sub.fd0 / sub.fh0))
        plt.xscale("log")
        plt.yscale("log")
        fig.savefig(
            TEST_OUTPUT_DIR / (f"test_random_state_calibrations_%s.pdf" % when), bbox_inches="tight"
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
    rt = None  # FL_rt_divoc_times()

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
        None,  # FL_test_rate_divoc_times(),
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
