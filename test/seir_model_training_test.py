import pathlib

import pytest
import pandas as pd
import numpy as np
import math
import structlog
from matplotlib import pyplot as plt

from pyseir.models.seir_model import (
    Demographics,
    SEIRModel,
    steady_state_ratios,
    NowcastingSEIRModel,
    ModelRun,
)

# rom pyseir.utils import get_run_artifact_path, RunArtifact
from test.mocks.inference import load_data
from test.mocks.inference.load_data import RateChange

TEST_OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output" / "test_results"


def make_tlist(num_days):
    return np.linspace(0, num_days, num_days + 1)


def create_standard_model(r0, sup, days=100, ratios=None):
    """
    Creates a standard model for testing purposes. Supports 
    - customization of r0 and suppresssion policy
    - optionally supply ratios for (non infection) compartments that matter
    """
    hosp_rate_general = 0.025
    initial_infected = 1000
    N = 10000000

    if ratios is None:
        model = SEIRModel(
            N=N,
            t_list=make_tlist(days),
            suppression_policy=sup,
            R0=r0,
            # gamma=0.7,
            # hospitalization_rate_general=hosp_rate_general,
            # hospitalization_rate_icu=0.3 * hosp_rate_general,
            A_initial=0.7 * initial_infected,
            I_initial=initial_infected,
            beds_general=N / 1000,
            beds_ICU=N / 1000,
            ventilators=N / 1000,
        )
    else:
        (E, I, A, HGen, HICU, HVent, D) = ratios
        model = SEIRModel(
            N=N,
            t_list=make_tlist(days),
            suppression_policy=sup,
            R0=r0,
            E_initial=E * initial_infected,
            A_initial=A * initial_infected,
            I_initial=initial_infected,
            HGen_initial=HGen * initial_infected,
            HICU_initial=HICU * initial_infected,
            HICUVent_initial=HVent * initial_infected,
            beds_general=N / 1000,
            beds_ICU=N / 1000,
            ventilators=N / 1000,
        )
    return model


def test_run_model_orig():
    def sup(t):
        return 1.0 if t < 50 else 0.6

    model = create_standard_model(1.4, sup, 200)

    model.run()

    fig = model.plot_results(alternate_plots=True)
    fig.savefig(TEST_OUTPUT_DIR / "test_run_model_orig.pdf", bbox_inches="tight")


def test_restart_existing_model_from_ratios():
    ratios = steady_state_ratios(1.4)

    def sup(t):
        return 1.0

    model = create_standard_model(1.4, sup, 50, ratios)
    model.run()

    fig = model.plot_results(alternate_plots=True)
    fig.savefig(TEST_OUTPUT_DIR / "test_restart_from_ratios.pdf", bbox_inches="tight")


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


def run_stationary(rt, t_over_x, x_is_new_cases=True):
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
            (ratios, ignore) = run_stationary(rt, t_over_x, x_is_new_cases)
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


def test_reproduce_FL_late_peak():
    """
    Reproduce behaviour of late peak in Florida where (as of ~Aug 1)
    - was 800 cases per day (20k tests per day with 4% positivity) steady for some time with 30 deaths per day
    - then R(t) went to 1.1 and 1.3 and 1.0 each for two weeks
    - deaths didn't rise (by 20%) until at least 4 weeks later -31-(-59)=28 than cases did
    - cases peaked at 120000/day and deaths at 185/day (2 weeks later)
    - with median age of recent cases = 37 years old
    """
    # new cases per day
    nC_initial = 800
    nC_max_exp = 12000.0
    nC_final_exp = 9600.0

    nD_max_exp = 185.0  # new deaths per day
    death_peak_delay_exp = 14  # days
    death_rising_start_delay = 28  # days
    # TODO change days below to be relative to Jan 1 not Aug 1

    def rt(t):
        adj = 0.0
        if t > -14:
            rt = 0.9
        elif t > -21:
            rt = 1.15
        elif t > -35:
            rt = 1.25
        elif t > -49:
            rt = 1.45
        elif t > -63:
            rt = 1.2
        else:
            rt = 1.0
        return rt + adj

    def test_rate(t):
        if t > -21:
            return 40000.0 - (25000.0 / 21.0) * t
        elif t > -28:
            return 50000.0 + (t + 28.0) / 7.0 * 15000.0
        elif t > -42:
            return 45000.0
        elif t > -56:
            return 30000.0
        else:
            return 20000.0 + (t + 63.0) / 7.0 * 10000.0

    # Generate reasonable starting ratios for all compartments
    (ignore, compartments) = run_stationary(1.0, t_over_x=20000.0 / nC_initial, x_is_new_cases=True)

    # Setup model and times
    model = NowcastingSEIRModel(median_age=37)
    t_list = np.linspace(-63, 0, 64)

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
        test_rate,
        rt,
        nC_initial=nC_initial,
        compartment_ratios_initial=compartments,
    )
    (results, ratios) = run.execute_dataframe_ratios()

    # TODO better check that people are conserved (see_model.py ModelRun._time_step)
    rN = results.tail(1).to_dict(orient="records")[0]
    r0 = results.head(1).to_dict(orient="records")[0]
    start = r0["S"] + r0["E"] + r0["I"] + r0["W"] + r0["C"] + r0["H"] + r0["D"] + r0["R"]
    end = rN["S"] + rN["E"] + rN["I"] + rN["W"] + rN["C"] + rN["H"] + rN["D"] + rN["R"]
    assert abs(start - end) < 1000.0

    # Check that new cases as expected
    assert rN["nC"] / nC_final_exp > 0.95 and rN["nC"] / nC_final_exp < 1.05

    # Check that daily deaths as expected - barely passing now
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

    # TODO generate plot like in our current charts


# TODO add a test for a recent peak with reasonably low test positivity
