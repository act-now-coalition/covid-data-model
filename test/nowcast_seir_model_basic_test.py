import os
import pathlib

import pytest  # pylint: disable=unused-import
import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt

from pyseir.models.demographics import Demographics, ContactsType
from pyseir.models.nowcast_seir_model import (
    extend_rt_function_with_new_cases_forecast,
    NowcastingSEIRModel,
    ModelRun,
)
from pyseir.models.historical_data import HistoricalData, adjust_rt_to_match_cases
from pyseir import OUTPUT_DIR

TEST_OUTPUT_DIR = pathlib.Path(os.path.join(OUTPUT_DIR, "test_results"))
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

MAKE_PLOTS = False  # Change to true to generate plots


def make_tlist(num_days):
    return np.linspace(0, num_days, num_days + 1)


def test_positivity_function():
    """
    Validate that the positivity function is continuous (in value and 1st derivative)
    and that it has the rigth behaviour for low and high values
    """
    if not MAKE_PLOTS:
        return
    t_over_i = np.array([math.pow(10.0, x) for x in np.linspace(-1.0, 2.0, 100)])
    model = NowcastingSEIRModel()

    positivity = np.array([model.positivity(ti) for ti in t_over_i])

    fig = plt.figure(facecolor="w", figsize=(20, 6))
    plt.subplot(121)
    plt.plot(t_over_i, positivity, label="Positivity")
    plt.xscale("log")
    plt.subplot(122)
    plt.plot(t_over_i, positivity * t_over_i, label="Positivity * T/I")
    plt.xscale("log")
    fig.savefig(TEST_OUTPUT_DIR / "test_positivity_function.pdf")


def test_median_to_age_fractions():
    """
    Test that Demographics.age_fraction_from_median generates reasonable yound, medium and old
    distributions for a wide range of values
    """
    # In Florida when median age was 37, 62% of cases were under 45 years old
    expected_young = 35.0 / 45.0 * 0.62
    distro_37 = Demographics.age_fractions_from_median(37.0)
    ratio = distro_37[0] / expected_young
    assert ratio > 0.95 and ratio < 1.05

    # FL during initial months
    distro_48 = Demographics.age_fractions_from_median(48.0)
    assert distro_48[2] > 0.25

    # BC during the initial months
    distro_53 = Demographics.age_fractions_from_median(53.0)
    assert distro_53[2] > 0.3

    # Push the other end
    distro_65 = Demographics.age_fractions_from_median(64.9)
    assert distro_65[2] > 0.45


def test_median_age_history():
    """
    Plot median age history for Florida so that it can be checked visually
    """
    if not MAKE_PLOTS:
        return
    t_list = t_list = np.linspace(80, 220, 200 - 80 + 1)
    f = Demographics.median_age_f("FL")
    m = [f(i) for i in t_list]
    fig, ax = plt.subplots()
    plt.plot(t_list, m)
    fig.savefig(TEST_OUTPUT_DIR / "test_median_age_history.pdf", bbox_inches="tight")


def test_validate_rt_over_time():
    """
    Check that our Bettencourt R(t) predictions integrate properly to explain
    the growth in new casees over time
    """
    t_list = np.linspace(100, 200, 200 - 100 + 1)

    results = []
    for state in HistoricalData.get_states():  # ["MI", "FL", "TX", "NY", "CA"]:
        if state == "VI":
            continue  # test data corrupt for this state
        (rt, nc, _1, _2, _3) = HistoricalData.get_state_data_for_dates(
            state, t_list, compartments_as_functions=True
        )

        (avg, adj, adj_rt) = adjust_rt_to_match_cases(rt, nc, t_list)
        (ignore1, check, ignore2) = adjust_rt_to_match_cases(adj_rt, nc, t_list)
        assert check > 0.95 and check < 1.05

        results.append((state, avg, adj))

    if not MAKE_PLOTS:
        return
    df = pd.DataFrame(results, columns=["state", "avg", "adj"])

    fig, ax = plt.subplots()
    ax.scatter(df.avg, df.adj)
    for i in df.index:
        ax.annotate(df["state"][i], (df["avg"][i], df["adj"][i]))

    plt.xlabel("Average R(t)")
    plt.ylabel("Case Ratio / Integrated R(t)")
    plt.yscale("log")
    fig.savefig(TEST_OUTPUT_DIR / "test_validate_rt_over_time.pdf", bbox_inches="tight")


def run_stationary(rt, median_age, t_over_x, x_is_new_cases=True):
    """
    Given R(t) and T/I or T/C run to steady state and return ratios of all compartments
    """

    model = NowcastingSEIRModel()
    x_fixed = 1000.0

    if x_is_new_cases:
        run = ModelRun(
            model,
            N=2e7,
            t_list=make_tlist(100),
            testing_rate_f=lambda t: t_over_x * x_fixed,
            rt_f=lambda t: rt,
            case_median_age_f=lambda t: median_age,
            initial_compartments={"nC": x_fixed},
            force_stationary=True,
        )
    else:
        i_fixed = 1000.0
        run = ModelRun(
            model,
            N=2e7,
            t_list=make_tlist(100),
            testing_rate_f=lambda t: t_over_x * x_fixed,
            rt_f=lambda t: rt,
            case_median_age_f=lambda t: median_age,
            initial_compartments={"I": x_fixed},
            force_stationary=True,
        )

    (history, ratios) = run.execute_lists_ratios()
    compartments = history[-1]
    ratios["rt"] = rt

    return (ratios, compartments)


def scan_rt(ratio, label, scales=(None, None), x_is_new_cases=True):
    """
    Check positivity function impact on various ratios by scanning R(t)
    at constant values of T (test rate) over x
    """
    if not MAKE_PLOTS:
        return
    fig = plt.figure(facecolor="w", figsize=(10, 6))
    for t_over_x in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        rows = list()
        for i in range(5, 25):
            rt = 0.1 * i
            (ratios, ignore) = run_stationary(rt, 38, t_over_x, x_is_new_cases)
            rows.append(ratios)

        df = pd.DataFrame(rows)

        line_label = "T/nC=%.2f" % t_over_x if x_is_new_cases else "T/I=%.2f" % t_over_x
        plt.plot(df["rt"], df[ratio], label=line_label)

    plt.xlabel("R(t)")
    plt.ylabel(label)
    if scales[1] is not None:
        plt.yscale(scales[1])
    if scales[1] is not None:
        plt.xscale(scales[0])
    plt.legend()
    fig.savefig(TEST_OUTPUT_DIR / ("test_scan_rt_%s.pdf" % ratio))


def test_scan_CFR():
    scan_rt("r_dD_nC", "new deaths / new cases", ("log", "log"), False)


def test_scan_test_fraction():
    scan_rt("r_C_IC", "test fraction", x_is_new_cases=True)


def test_historical_peaks_positivity_to_real_cfr():
    """
    Illustrate dependence between peaks in deaths and cases (ratio) as a 
    function of positivity
    """
    if not MAKE_PLOTS:
        return
    peaks = pd.read_csv("test/data/historical/historical_peaks.csv")
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


def test_reproduce_FL_demographic_shift():
    """
    Florida is the one state where we have median age data. Test that this median age time history
    can be recreated using only R(t) and median age history for US as a whole - a process we can
    use with any state
    """
    t_list = np.linspace(120, 230, 230 - 120 + 1)

    expected_f = Demographics.median_age_f("FL")
    default_usa_f = Demographics.median_age_f()
    demo = Demographics(median_age=expected_f(t_list[0]))

    (rt_f, ignore1, ignore2, ignore3, ignore4) = HistoricalData.get_state_data_for_dates(
        "FL", t_list
    )

    values = [demo.get_median_age()]
    for t in t_list[:-1]:
        demo.evolve_median(rt_f(t), default_usa_f(t))
        values.append(demo.get_median_age())

    if not MAKE_PLOTS:
        return
    fig = plt.figure(facecolor="w", figsize=(10, 6))
    plt.plot(t_list, values, label="results")
    plt.plot(t_list, [expected_f(t) for t in t_list], label="expected")
    plt.legend()
    fig.savefig(TEST_OUTPUT_DIR / "test_reproduce_FL_demographic_shift.pdf")


def test_using_outputs_of_case_forecast_to_extend_rt():
    """
    Demonstrates how to use the output of a new case forecast in the future, along with historical
    R(t) function to generate an extended R(t) function out into the future
    """

    future = 250
    t_list = np.linspace(120, future, future - 120 + 1)

    for state in ["AK", "AZ", "FL", "HI", "IL", "NY", "WI"]:

        # Using each state's data to use as forecast cases below and to roughly check extended R(t)
        # function looks reasonable
        (rt_f, nC_f, ignore2, ignore3, ignore4) = HistoricalData.get_state_data_for_dates(
            state, t_list, compartments_as_functions=True
        )

        # As test take forecasted cases (past 140) from historical data for new cases in Illinois
        forecasted_cases = []
        for day in [140, 160, 180, 200, 220]:
            forecasted_cases.append((day, nC_f(day)))
        start = forecasted_cases[0][0]
        end = forecasted_cases[-1][0]

        # Now generate an extended rt_f function from
        # - the forecasted new cases at various times in the future
        # - the rt_f from Bettencourt for the same source data (Illinois)
        serial_period = NowcastingSEIRModel().serial_period
        forecast_rt_f = extend_rt_function_with_new_cases_forecast(
            rt_f, serial_period, forecasted_cases
        )

        # Check result for the final new cases at end of extended R(t)
        check_nC = [nC_f(start)]
        check_t = list(range(start, future))
        for t in check_t[1:]:
            check_nC.append(check_nC[-1] * math.exp((forecast_rt_f(t) - 1) / serial_period))

        if MAKE_PLOTS:
            # Plot resulting R(t), cases and compare with Bettencourt R(t), actual cases
            fig = plt.figure(facecolor="w", figsize=(8, 8))
            fig.suptitle((f"Does R(t) extrapolation fit cases for %s?" % state))

            plt.subplot(211)
            plt.ylabel("R(t)")
            plt.plot(t_list, [forecast_rt_f(t) for t in t_list], label="piecewise linear R(t)")
            plt.plot(t_list, [rt_f(t) for t in t_list], label="Bettencourt R(t)", linestyle="--")
            for (day, nc) in forecasted_cases:
                plt.plot([day, day], [0.5, 1.5], linestyle="--", color="black")
            plt.legend()

            plt.subplot(212)
            plt.ylabel("New cases")
            plt.plot(check_t, check_nC, label="from piecewise linear R(t)")
            plt.plot(t_list, [nC_f(i) for i in t_list], linestyle="--", label="actual cases")
            plt.yscale("log")
            plt.legend()

            fig.savefig(
                TEST_OUTPUT_DIR
                / (f"test_using_outputs_of_case_forecast_to_extend_rt_%s.pdf" % state)
            )

        # Check that cases match
        nC_ratio = check_nC[-(future - end)] / nC_f(end)
        assert nC_ratio > 0.95 and nC_ratio < 1.05


def test_scipy_interpolate():
    from scipy.interpolate import interp1d

    if not MAKE_PLOTS:
        return

    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-(x ** 2) / 9.0)
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind="cubic")

    xnew = np.linspace(0, 10, num=41, endpoint=True)

    fig = plt.figure(facecolor="w", figsize=(8, 8))
    plt.plot(x, y, "o", xnew, f(xnew), "-", xnew, f2(xnew), "--")
    plt.legend(["data", "linear", "cubic"], loc="best")
    fig.savefig(TEST_OUTPUT_DIR / "test_scipy_interpolate.pdf", bbox_inches="tight")


################################ Obsolete test cases ############################


def obsolete_test_evolve_median_age():
    """
    Considering having model equations to evolve the median age of new cases forward
    in time rather than requiring it to be an input. Did something simpler but may come
    back to this code later
    """
    pop = Demographics(0.4, 0.4, 0, 0.0)  # general population
    inf = Demographics(0.2, 0.5, 1000, 6.0)  # infected
    results = [inf.as_array()]
    for i in range(1, 100):
        c = ContactsType.RESTRICTED if i < 50 else ContactsType.LOCKDOWN
        inf.update_by_contacting_another(c, 140, pop)
        results.append(inf.as_array())
    bins = ["young", "medium", "old"]
    df = pd.DataFrame(results, columns=bins)
    fig, ax = plt.subplots()
    for bin in bins:
        plt.plot(df.index, df[bin], label=bin)
    fig.legend()
    fig.savefig(TEST_OUTPUT_DIR / "test_evolve_median_age.pdf", bbox_inches="tight")
