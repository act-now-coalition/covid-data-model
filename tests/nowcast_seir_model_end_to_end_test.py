import pathlib
import math
from datetime import datetime

import pytest  # pylint: disable=unused-import
import pandas as pd
import numpy as np

from scipy import stats
from matplotlib import pyplot as plt

from pyseir.models.demographics import Demographics
from pyseir.models.nowcast_seir_model import (
    ramp_function,
    NowcastingSEIRModel,
    ModelRun,
    extend_rt_function_with_new_cases_forecast,
)

from pyseir.models.historical_data import (
    HistoricalData,
    ForecastData,
    adjust_rt_to_match_cases,
    EARLY_OUTBREAK_START_DAY_BY_STATE,
)

from pyseir import OUTPUT_DIR

TEST_OUTPUT_DIR = pathlib.Path(OUTPUT_DIR) / "test_results"

MAKE_PLOTS = False  # Change to true to generate plots


def test_run_new_model_incrementally():
    """
    Demonstrates how to minimally run the new model incrementally from an initial set of
    observables (new cases, hospitalizations, new deaths) and with given assumptions (for
    r(t), test_rate and median_age) into the future.
    """

    # Run into the future using R(t) ramp starting sometime in the past
    (start, today, ramp_end, future) = (200, 230, 280, 320)
    t_list = np.linspace(start, future, future - start + 1)

    # Try changing these to get different possible futures
    nC_ramp_to = 8000.0
    nC_future = 15000.0

    # Need assumptions for R(t) and testing_rate(t) for the future as inputs
    data_tlist = np.linspace(start, today, today - start + 1)

    # This would really be smoothed current values
    (rt, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates("FL", data_tlist)

    # Create extended R(t) function using projected cases sometime in the future
    forecast_rt_f = extend_rt_function_with_new_cases_forecast(
        rt,
        NowcastingSEIRModel().serial_period,
        [(start, nc[start]), (today, nc[today]), (ramp_end, nC_ramp_to), (future, nC_future)],
    )

    start = datetime.now()
    with_history = True  # try True or False

    # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
    run = ModelRun(
        NowcastingSEIRModel(),  # creating a default model here with no trained parameters
        20e6,  # N, population of jurisdiction
        t_list,
        None,  # tests (these are currently ignored so positivity not used)
        forecast_rt_f,
        case_median_age_f=Demographics.infer_median_age_function(t_list, forecast_rt_f),
        historical_compartments={"nC": nc, "H": h, "nD": nd} if with_history else None,
        initial_compartments=None
        if with_history
        else {"nC": nc[start], "H": h[start], "nD": nd[start]},
        auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
        auto_calibrate=True,  # Override some model constants to ensure continuity with initial compartments
    )

    # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
    (results, ratios, fig) = run.execute_dataframe_ratios_fig()

    # Should have finished in less that 1 second
    elapsed = (datetime.now() - start).seconds

    if MAKE_PLOTS:
        fig.savefig(TEST_OUTPUT_DIR / "test_run_new_model_incrementally.pdf")

    assert elapsed < 1


def test_historical_peaks_positivity_to_real_cfr():
    """
    Shows the dependency between test positivity and CFR by looking at historical
    data peaks of various states over time
    """
    if MAKE_PLOTS:
        return

    peaks = pd.read_csv("tests/data/historical/historical_peaks.csv")
    early_peaks = peaks[peaks["when"] == "Apr-May"].copy()
    late_peaks = peaks[peaks["when"] == "Jun-Jul"].copy()

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


def test_demonstrate_hospitalization_delay_changes():
    """
    Demonstrates that hospitalization peaks are delayed relative to new cases
    by varying number of days depending on how well states were prepared at
    various points in time
    """

    if MAKE_PLOTS:
        return

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
        plt.xlabel(f"Hospitalizations (delayed %s days)" % s["delay"])
        plt.ylabel("new Deaths")
        plt.yscale("log")
        plt.xscale("log")
        # plt.ylim((0.01, 1.0))
        fig.legend()
        fig.savefig(TEST_OUTPUT_DIR / (f"test_ratio_evolution_%s_states.pdf" % name))


@pytest.mark.slow
def test_historical_period_state_deaths_and_hospitalizations():
    """
    Validate model by recreating historical death and hospitalization timeseries starting
    for specific periods in time from initial conditions and given correct R(t) as input.
    TODO refactor so can have several separate tests with same structure
    TODO add many tests for one state at different times
    TODO get most recent data from Brett and retest recent
    """

    # For recent tests with states we should have this level of accuracy
    # Note smape in [0.,2.] - best values for recent avg is about .335
    TARGET_SMAPE = 0.35

    states = HistoricalData.get_states()
    starts = {"early": 90, "recent": 155}  # calendar day in 2020 when each test starts
    num_days = 95  # duration of each test

    # Various state outbreaks started at different points in time, overrides starts.early
    early_start = EARLY_OUTBREAK_START_DAY_BY_STATE

    # Keep track of the model calibrations for each state and period
    calibrations = []

    for state in states:
        # Do a test for each period (when) for each state
        for (when, std_start) in starts.items():
            earliest = (
                early_start[state] if (when == "early" and state in early_start) else std_start
            )
            start = earliest
            t_list = np.linspace(start, start + num_days, num_days + 1)
            t0 = t_list[0]

            # Adjusting for change in delay over time
            # h_delay = 3.0 if when == "early" else 10.0
            h_delay = 0.0

            # Get historical data for that state and adjust R(t) for long term bias
            (rt, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates(state, t_list)
            if len(nc) != len(t_list):  # no data returned
                continue
            (average_R, growth_ratio, adj_r_f) = adjust_rt_to_match_cases(
                rt, lambda t: nc[t], t_list
            )

            # Confiigure and run the model with history for initial constraints and to add actuals to charts
            run = ModelRun(
                NowcastingSEIRModel(),  # delay_ci_h=h_delay, death_delay=0),
                20e6,  # N
                t_list,
                tests,
                adj_r_f,
                case_median_age_f=Demographics.infer_median_age_function(
                    t_list, rt
                ),  # Demographics.median_age_f(state),
                historical_compartments={"nC": nc, "H": h, "nD": nd},
                auto_initialize_other_compartments=True,
                auto_calibrate=True,  # True
            )
            (results, ratios, fig) = run.execute_dataframe_ratios_fig(plot=MAKE_PLOTS)
            calibrations.append(
                (state, when, run.model.lr_fh[0], run.model.lr_fd[0], ratios["SMAPE"],)
            )

            if MAKE_PLOTS and when == "recent":
                # Save figure for each state in each period
                # TODO store test result metrics in file rather than only in chart title string
                fig.savefig(
                    (
                        TEST_OUTPUT_DIR
                        / (
                            f"test_historical_period_%s_period_%s_start=%d_calibration=(%.2f,%.2f->%.2f)_smape=%.2f.pdf"
                            % (
                                when,
                                state,
                                start,
                                run.model.lr_fh[0],
                                run.model.lr_fd[0],
                                run.model.lr_fh[0] * run.model.lr_fd[0],
                                ratios["SMAPE"],
                            )
                        )
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig)

    # Create summary chart for each "period" showing calibration and SMAPE of all states
    df = pd.DataFrame(calibrations, columns=["state", "when", "fh0", "fd0", "smape"])
    df["markersize"] = (10.0 * df["smape"] + 1.0).astype(int)

    for when in ["early", "recent"]:
        sub = df[df["when"] == when]
        fig, ax = plt.subplots()
        rect = plt.Rectangle([0.5, 0.5], 1.5, 1.5, facecolor="g", alpha=0.2)
        ax.add_patch(rect)

        for i in sub.index:
            plt.plot([sub.fh0[i],], [sub.fd0[i],], "o", markersize=sub.markersize[i], color="b")
            ax.annotate(sub["state"][i], (sub["fh0"][i], sub["fd0"][i]))

        plt.xlabel("fh0 (gmean =%.2f)" % stats.gmean(sub.fh0))
        plt.ylabel("fd0 (gmean =%.2f)" % stats.gmean(sub.fd0))
        plt.xscale("log")
        plt.yscale("log")
        plt.title(f"%s with avg SMAPE=%.3f" % (when, sub["smape"].mean()))
        fig.savefig(
            (TEST_OUTPUT_DIR / (f"test_historical_period_%s_state_calibrations.pdf" % when)),
            bbox_inches="tight",
        )
        plt.close(fig)

        # Validate that accuracy of recent period state values is still beating the threshold
        if when == "recent":
            assert sub["smape"].mean() < TARGET_SMAPE

        # TODO add more assertions on a per state level


@pytest.mark.slow
def test_multiple_periods_for_select_states():
    """
    Show how fit evolves over time for one staste
    """

    for state in [
        "AZ",
        "CT",
        "FL",
        "GA",
        "IA",
        "LA",
        "MA",
        "ME",
        "NE",
        "MO",
        "MT",
        "NY",
        "OR",
        "TX",
        "VA",
    ]:

        latest = 250
        # Get actuals to compare to
        t_all = np.linspace(100, latest, latest - 100 + 1)
        (rt, nc_all, ignore3, h_all, nd_all) = HistoricalData.get_state_data_for_dates(state, t_all)

        if MAKE_PLOTS:
            # Create chart
            fig = plt.figure(facecolor="w", figsize=(12, 7))
            plt.title(f"State = %s" % state)
            plt.plot(t_all, nd_all, label="actual nD")
            plt.plot(t_all, h_all / 10.0, label="actual H/10")

        for today in [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]:
            duration = min(30, latest - today)
            t_list = np.linspace(today, today + duration, duration + 1)

            # Need assumptions for R(t) and testing_rate(t) for the future as inputs
            (ignore, nc, tests, h, nd) = HistoricalData.get_state_data_for_dates(state, t_list)

            # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
            model = NowcastingSEIRModel()
            run = ModelRun(
                model,  # creating a default model here with no trained parameters
                20e6,  # N, population of jurisdiction
                t_list,
                None,  # tests (these are currently ignored so positivity not used)
                rt,
                case_median_age_f=Demographics.infer_median_age_function(t_list, rt),
                initial_compartments={"nC": nc[today], "H": h[today], "nD": nd[today]},
                auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
                auto_calibrate=True,  # Override some model constants to ensure continuity with initial compartments
            )

            # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
            (results, ignore, ignore) = run.execute_dataframe_ratios_fig(plot=False)

            (fh0, fd0) = model.get_calibration()
            if MAKE_PLOTS:
                plt.plot(
                    t_list,
                    results["nD"].values,
                    label=(f"t=%d: (%.2f, %.2f)" % (today, fh0, fd0)),
                    linestyle="--",
                    color="grey",
                )
                plt.plot(t_list, results["H"].values / 10.0, linestyle="--", color="grey")

        if MAKE_PLOTS:
            plt.legend()
            plt.xlim(70, latest)
            plt.yscale("log")
            fig.savefig(
                TEST_OUTPUT_DIR / (f"test_multiple_periods_for_select_states_%s.pdf" % state)
            )
            plt.close(fig)


@pytest.mark.slow
def test_reproduce_TX_late_peak():
    """
    Reproduce behaviour of late peaks for Texas starting from t0 = 170
    - peak H=1050 at t=200
    - peak nD=200 at t=220

    TODO move over assertions from the Florida case that no longer runs
    TODO filter the data to remove the high value for texas on about t=210
    """

    if not MAKE_PLOTS:
        return

    # Time period for the run (again relative to Jan 1, 2020)
    t_list = np.linspace(
        150, 230, 230 - 150 + 1
    )  # Here starts from when FL started to ramp up cases

    # Need assumptions for R(t) and testing_rate(t) for the future as inputs
    (rt, nC, tests, H, nD) = HistoricalData.get_state_data_for_dates("VA", t_list)
    # Here taken from historical data but typically will use R(t) projections

    # Infer median age function for Texas
    median_age_f = Demographics.infer_median_age_function(t_list, rt)
    # Does slightly better with Demographics.median_age_f("FL")

    # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
    run = ModelRun(
        NowcastingSEIRModel(
            # delay_ci_h=5
        ),  # creating a default model here with no trained parameters
        20e6,  # N, population of jurisdiction
        t_list,
        tests,
        rt,
        case_median_age_f=median_age_f,
        historical_compartments={"nC": nC, "H": H, "nD": nD},
        auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
        auto_calibrate=True,  # Override some model constants to ensure continuity with initial compartments
    )

    # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
    (results, ratios, fig) = run.execute_dataframe_ratios_fig()
    fig.savefig(TEST_OUTPUT_DIR / "test_reproduce_TX_late_peak.pdf", bbox_inches="tight")


@pytest.mark.slow
def test_inertia_of_model():
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
        (results, ratios, fig) = run.execute_dataframe_ratios_fig(plot=MAKE_PLOTS)

        nC_peak_at = results["nC"].argmax()
        h_peak_at = results["H"].argmax()
        nD_peak_at = results["nD"].argmax()

        h_delay = h_peak_at - nC_peak_at
        d_delay = nD_peak_at - h_peak_at

        h_to_nC = results["H"][h_peak_at] / results["nC"][nC_peak_at]
        nD_to_h = results["nD"][nD_peak_at] / results["H"][h_peak_at]

        delays.append([t_i, model.t_h(48), h_delay, h_to_nC, d_delay, nD_to_h])
        assert h_delay - int(t_i + model.t_h(48) / 2) in [-2, -1, 0, 1, 2, 3]  # 0 +/- 1
        assert (d_delay) in [2, 1, 0]  # 1 +/- 1

    if MAKE_PLOTS:
        fig.savefig(TEST_OUTPUT_DIR / "test_inertia_of_model.pdf", bbox_inches="tight")


@pytest.mark.slow
def test_simulate_iowa_late_august():
    """
    What could happen in Iowa given similar behaviour as in Florida?
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
    (results, ratios, fig) = run.execute_dataframe_ratios_fig(plot=MAKE_PLOTS)

    results.to_csv(TEST_OUTPUT_DIR / "test_simulate_iowa_results.csv")
    if MAKE_PLOTS:
        fig.savefig(TEST_OUTPUT_DIR / "test_simulate_iowa.pdf", bbox_inches="tight")


@pytest.mark.slow
def test_reproducing_forecast_deaths_from_cases():
    """
    Use forecast cases from Reich Labs and evolve forward in time from present
    using that data determining new deaths. Compare those with values contained in
    forecast
    """
    for state in ["FL", "GA", "TX", "DE", "WI"]:

        today = 250  # TODO compute for today

        # Get actuals to compare to
        t_history = np.linspace(today - 30, today, 30 + 1)
        (rt, nc, ignore_tests, h, nd) = HistoricalData.get_state_data_for_dates(state, t_history)

        # Get latest forecast data
        (forecast_nC, forecast_nD) = ForecastData.get_state_nC_nD_forecast(state)
        now_and_forecast = [(today, nc[today])] + forecast_nC

        latest = now_and_forecast[-1][0]  # Latest day in the forecast
        t_list = np.linspace(today, latest, latest - today + 1)

        # Create extended R(t) function using forecasted_cases
        forecast_rt_f = extend_rt_function_with_new_cases_forecast(
            rt, NowcastingSEIRModel().serial_period, now_and_forecast,
        )

        # Create a ModelRun (with specific assumptions) based on a Model (potentially with some trained inputs)
        model = NowcastingSEIRModel()
        run = ModelRun(
            model,  # creating a default model here with no trained parameters
            20e6,  # N, population of jurisdiction
            t_list,
            None,  # tests (these are currently ignored so positivity not used)
            forecast_rt_f,
            case_median_age_f=Demographics.infer_median_age_function(t_list, forecast_rt_f),
            initial_compartments={"nC": nc[today], "H": h[today], "nD": nd[today]},
            auto_initialize_other_compartments=True,  # By running to steady state at initial R(t=t0)
            auto_calibrate=True,  # Override some model constants to ensure continuity with initial compartments
        )

        # Execute the run, results are a DataFrame, fig is Matplotlib Figure, ratios are key metrics
        (results, ignore, fig) = run.execute_dataframe_ratios_fig(plot=MAKE_PLOTS)

        if not MAKE_PLOTS:
            return

        fig.savefig(
            TEST_OUTPUT_DIR / (f"test_reproducing_forecast_deaths_from_cases_%s_run.pdf" % state),
            bbox_inches="tight",
        )
        plt.close(fig)

        # Now compare results against forecasts and for continuity with actuals
        run_nd = results.nD

        fig = plt.figure(facecolor="w", figsize=(10, 7))
        plt.title(f"Compare deaths with forecast for state = %s" % state)
        plt.scatter(t_history, nd, label="actual data", marker="o")
        plt.scatter(
            [forecast_nD[i][0] for i in range(0, len(forecast_nD))],
            [forecast_nD[i][1] for i in range(0, len(forecast_nD))],
            marker="o",
            label="Reich Lab forecast",
        )
        plt.plot(t_list, run_nd.values, label="run results")
        plt.legend()

        fig.savefig(
            TEST_OUTPUT_DIR
            / (f"test_reproducing_forecast_deaths_from_cases_%s_compare.pdf" % state),
            bbox_inches="tight",
        )
        plt.close(fig)


############################### Obsolete tests below here ##############################


@pytest.mark.slow
def obsolete_test_reproduce_FL_late_peak():
    """
    This test is obsolete.
    TODO move its assertions over to the Texas case just above
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
    rt = lambda t: t  # FL_rt_divoc_times()

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
