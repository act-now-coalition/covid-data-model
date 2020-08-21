import pathlib

import pytest
import pandas as pd
import numpy as np
import math
from random import choices, randrange
import structlog
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

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


def make_tlist(num_days):
    return np.linspace(0, num_days, num_days + 1)


def test_positivity_function():
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
    t_list = t_list = np.linspace(80, 220, 200 - 80 + 1)
    f = Demographics.median_age_f("FL")
    m = [f(i) for i in t_list]
    fig, ax = plt.subplots()
    plt.plot(t_list, m)
    fig.savefig(TEST_OUTPUT_DIR / "test_median_age_history.pdf", bbox_inches="tight")


def test_validate_rt_over_time():
    t_list = np.linspace(100, 200, 200 - 100 + 1)

    results = []
    for state in HistoricalData.get_states():  # ["MI", "FL", "TX", "NY", "CA"]:
        (rt, nc, _1, _2, _3) = HistoricalData.get_state_data_for_dates(
            state, t_list, as_functions=True
        )

        (avg, adj, adj_rt) = adjust_rt_to_match_cases(rt, nc, t_list)
        (ignore1, check, ignore2) = adjust_rt_to_match_cases(adj_rt, nc, t_list)
        assert check > 0.95 and check < 1.05

        results.append((state, avg, adj))
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
            nC_initial=x_fixed,
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
            I_initial=x_fixed,
            force_stationary=True,
        )

    (history, ratios) = run.execute_lists_ratios()
    compartments = history[-1]
    ratios["rt"] = rt

    return (ratios, compartments)


def scan_rt(ratio, label, scales=(None, None), x_is_new_cases=True):
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

