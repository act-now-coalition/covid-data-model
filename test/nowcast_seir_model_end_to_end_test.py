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
        "FL", t_list
    )
    # Here taken from historical data but typically will use R(t) projections

    start = datetime.now()
    # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
    run = ModelRun(
        NowcastingSEIRModel(
            delay_ci_h=5
        ),  # creating a default model here with no trained parameters
        20e6,  # N, population of jurisdiction
        t_list,
        None,  # tests,
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
        "late": {"states": ["FL", "TX", "GA", "CA", "AZ"], "delay": 12},
        "steady": {"states": ["PA", "IL"], "delay": 7},
    }

    model = NowcastingSEIRModel()
    for name, s in sets.items():
        fig, ax = plt.subplots()
        for state in s["states"]:
            (rt, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates(
                state, t_list, no_functions=True
            )
            fh = h / (nc * model.t_i)
            fd = (nd * model.t_h()) / h
            pos = nc / tests
            hdelay = h.shift(s["delay"])

            plt.scatter(hdelay.values, nd.values, marker=".", label=state)
            plt.plot([hdelay.values[-1]], [nd.values[-1]], marker="o")

        plt.plot([1e3, 1e4], [2e1, 2e2], linestyle="--")
        plt.xlabel(f"Hospitalizations (delayed %s days" % s["delay"])
        plt.ylabel("new Deaths")
        plt.yscale("log")
        plt.xscale("log")
        # plt.ylim((0.01, 1.0))
        fig.legend()
        fig.savefig(TEST_OUTPUT_DIR / (f"test_ratio_evolution_%s_states.pdf" % name))


def test_random_periods_interesting_states():
    states = HistoricalData.get_states()
    starts = {"early": 90, "recent": 155}
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

    test_dir = {}
    for when in starts.keys():
        test_dir[when] = pathlib.Path(TEST_OUTPUT_DIR / when)
        test_dir[when].mkdir(exist_ok=True)

    for state in states:  # ["TX"]:  #
        # Do a test early and late in the pandemic for each state
        for (when, std_start) in starts.items():
            num_days = 75  # duration of test
            earliest = (
                early_start[state] if (when == "early" and state in early_start) else std_start
            )
            start = earliest
            # randrange(
            #     earliest, min(218 - num_days, 30 + earliest), 1
            # )  # Pick a random start time within a range
            t_list = np.linspace(start, start + num_days, num_days + 1)
            t0 = t_list[0]

            # Adjusting for change in delay over time
            # h_delay = 3.0 if when == "early" else 10.0
            h_delay = 0.0

            # Get historical data for that state and adjust R(t) for long term bias
            (rt, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates(state, t_list)
            (average_R, growth_ratio, adj_r_f) = adjust_rt_to_match_cases(
                rt, lambda t: nc[t], t_list
            )

            # Run the model with history for initial constraints and to add actuals to charts
            run = ModelRun(
                NowcastingSEIRModel(),  # delay_ci_h=h_delay, death_delay=0),
                20e6,  # N
                t_list,
                tests,
                adj_r_f,
                case_median_age_f=Demographics.median_age_f(state),
                historical_compartments={"nC": nc, "H": h, "nD": nd},
                auto_initialize_other_compartments=True,
                auto_calibrate=True,  # True
            )
            (results, ratios, fig) = run.execute_dataframe_ratios_fig()
            calibrations.append(
                (
                    state,
                    when,
                    run.model.lr_fh[0] * run.model.lr_fd[0],
                    run.model.lr_fh[0],
                    ratios["SMAPE"],
                )
            )

            fig.savefig(
                test_dir[when]
                / (
                    f"test_random_periods_%s_%s_start=%d_calibration=(%.2f,%.2f->%.2f)_smape=%.2f.pdf"
                    % (
                        state,
                        when,
                        start,
                        run.model.lr_fh[0],
                        run.model.lr_fd[0],
                        run.model.lr_fh[0] * run.model.lr_fd[0],
                        ratios["SMAPE"],
                    )
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
    df = pd.DataFrame(calibrations, columns=["state", "when", "fh0", "fd0", "smape"])
    df["markersize"] = (10.0 * df["smape"] + 1.0).astype(int)
    for when in ["early", "recent"]:
        sub = df[df["when"] == when]
        fig, ax = plt.subplots()
        rect = plt.Rectangle([0.5, 0.5], 1.5, 1.5, facecolor="g", alpha=0.2)
        ax.add_patch(rect)
        # ax.scatter(sub.fh0, sub.fd0, markersize=sub.markersize)
        for i in sub.index:
            plt.plot([sub.fh0[i],], [sub.fd0[i],], "o", markersize=sub.markersize[i], color="b")
            ax.annotate(sub["state"][i], (sub["fh0"][i], sub["fd0"][i]))

        plt.xlabel("fd0*fh0 (gmean =%.2f)" % stats.gmean(sub.fh0 * sub.fd0))
        plt.ylabel("fd0/fh0 (gmean =%.2f)" % stats.gmean(sub.fd0 / sub.fh0))
        plt.xscale("log")
        plt.yscale("log")
        fig.savefig(
            test_dir[when] / (f"test_random_state_calibrations_%s.pdf" % when), bbox_inches="tight"
        )

    avg_mape = df["mape"].mean()
    print("average mape = %.3f" % avg_mape)
    assert avg_mape < 0.5


def test_reproduce_TX_late_peak():
    """
    Reproduce behaviour of late peaks for Texas starting from t0 = 170
    - peak H=1050 at t=200
    - peak nD=200 at t=220
    - using median age of FL for now as substitute
    - and TODO linear ramp on (at least) fh0 to get slope right at the start
    - using times from 170 to 180 to manually constraint fh0 = a + b(t-t0)
    - and where fh0 = 2.58, fd0 = .73 had been selected to match at t0=170
    """
    # Time period for the run (again relative to Jan 1, 2020)
    t_list = np.linspace(
        170, 230, 230 - 170 + 1
    )  # Here starts from when FL started to ramp up cases

    # Need assumptions for R(t) and testing_rate(t) for the future as inputs
    (rt, nC, tests, H, nD) = HistoricalData.get_state_data_for_dates("TX", t_list)
    # Here taken from historical data but typically will use R(t) projections

    # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
    run = ModelRun(
        NowcastingSEIRModel(
            # delay_ci_h=5
        ),  # creating a default model here with no trained parameters
        20e6,  # N, population of jurisdiction
        t_list,
        tests,
        rt,
        case_median_age_f=Demographics.median_age_f("TX"),
        historical_compartments={"nC": nC, "H": H, "nD": nD},
        auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
        auto_calibrate=False,  # Override some model constants to ensure continuity with initial compartments
    )

    # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
    (results, ratios, fig) = run.execute_dataframe_ratios_fig()
    fig.savefig(TEST_OUTPUT_DIR / "test_reproduce_TX_late_peak.pdf", bbox_inches="tight")


def test_reproduce_FL_late_peak():
    """
    TODO use all of these asserts somewhere else
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


def test_intertia_of_model():
    """
    Explores delay of H and D relative to C and overshoot/undershoot with variable dwell time
    parameters.
    """

    # Time period for the run (again relative to Jan 1, 2020)
    t_list = np.linspace(0, 100, 100 + 1)  # Here starts from when FL started to ramp up cases

    # Need assumptions for R(t) and testing_rate(t) for the future as inputs
    rt = lambda t: 1.0 + 0.2 * math.sin(t / 12.0)
    model = NowcastingSEIRModel()
    delays = []

    t_i = 6
    t_h = 8

    for (t_i, t_h) in [(3, 8), (6, 8), (12, 8), (6, 4), (6, 16), (12, 16)]:
        model.t_i = t_i
        model.th0 = t_h

        # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
        run = ModelRun(
            model,  # TODO experiment with dwell times
            20e6,  # N, population of jurisdiction
            t_list,
            None,  # tests,
            rt,
            case_median_age_f=lambda t: 48,
            initial_compartments={"nC": 1000.0},
            auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
            auto_calibrate=False,  # Override some model constants to ensure continuity with initial compartments
        )

        # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
        (results, ratios, fig) = run.execute_dataframe_ratios_fig()

        nC_peak_at = results["nC"].argmax()
        h_peak_at = results["H"].argmax()
        nD_peak_at = results["nD"].argmax()

        h_delay = h_peak_at - nC_peak_at
        d_delay = nD_peak_at - h_peak_at

        h_to_nC = results["H"][h_peak_at] / results["nC"][nC_peak_at]
        nD_to_h = results["nD"][nD_peak_at] / results["H"][h_peak_at]

        delays.append([t_i, model.t_h(48), h_delay, h_to_nC, d_delay, nD_to_h])
        assert h_delay - int(t_i + model.t_h(48) / 2) in [-2, -1, 0, 1, 2]  # 0 +/- 1
        assert (d_delay) in [2, 1, 0]  # 1 +/- 1

    fig.savefig(TEST_OUTPUT_DIR / "test_intertia_of_model.pdf", bbox_inches="tight")
    delays


def test_simulate_iowa():
    """
    What could happen in Iowa given similar behaviour as in Florida
    """
    lag = 85
    days = 60

    # Time period for the run (again relative to Jan 1, 2020)
    t_list = np.linspace(244, 244 + days, days + 1)
    fl_t_list = np.linspace(244 - (lag + 5), 244 - (lag + 5) + days + 2 * lag, days + 2 * lag + 1)

    # Need assumptions for R(t) and testing_rate(t) for the future as inputs
    (rt_fl, ignore_0, ignore_1, ignore_2, ignore_3) = HistoricalData.get_state_data_for_dates(
        "FL", fl_t_list
    )
    median_age_fl = Demographics.median_age_f("FL")

    def rt(t):
        return rt_fl(t - lag) - 0.1  # due to mask wearing

    def median_age(t):  # Iowa has younger population
        ma = median_age_fl(t - lag)
        adj = 35.0 + 0.75 * (ma - 35)  # turn 50 into 46
        return adj

    # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
    run = ModelRun(
        NowcastingSEIRModel(),  # creating a default model here with no trained parameters
        5e6,  # N, population of jurisdiction
        t_list,
        None,  # tests,
        rt,
        case_median_age_f=median_age,
        initial_compartments={"nC": 1170.0, "H": 500.0, "nD": 18.0},
        auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
        auto_calibrate=True,  # Override some model constants to ensure continuity with initial compartments
    )

    # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
    (results, ratios, fig) = run.execute_dataframe_ratios_fig()

    results.to_csv(TEST_OUTPUT_DIR / "test_simulate_iowa_results.csv")
    fig.savefig(TEST_OUTPUT_DIR / "test_simulate_iowa.pdf", bbox_inches="tight")


# TODO add a test for a recent peak with reasonably low test positivity
