from datetime import timedelta
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from pyseir.load_data import HospitalizationDataType  # Do we still need this?
from pyseir.utils import get_run_artifact_path, RunArtifact


def plot_fitting_results(result) -> None:
    """

    """
    # Entry point from model_fitter
    output_file = get_run_artifact_path(result.fips, RunArtifact.MLE_FIT_REPORT)

    # Save the mle fitter
    fig = result.mle_model.plot_results()
    fig.savefig(output_file.replace("mle_fit_results", "mle_fit_model"), bbox_inches="tight")

    # Save the model in log mode
    fig = _plot_ModelFitter_results(result)
    fig.gca().set_yscale("log")
    fig.savefig(output_file, bbox_inches="tight")

    # Save the model in linear mode
    fig = _plot_ModelFitter_results(result)
    fig.gca().set_yscale("linear")

    fig.savefig(
        output_file.replace("mle_fit_results", "mle_fit_results_linear"), bbox_inches="tight"
    )
    return


def _plot_ModelFitter_results(result) -> plt.Figure:
    """
    Take a model_fitter.ModelFitter instance and produce the plots.
    """
    data_dates = [result.ref_date + timedelta(days=t) for t in result.times]
    if result.hospital_times is not None:
        hosp_dates = [result.ref_date + timedelta(days=float(t)) for t in result.hospital_times]
    if result.icu_times is not None:
        icu_dates = [result.ref_date + timedelta(days=float(t)) for t in result.icu_times]

    model_dates = [
        result.ref_date + timedelta(days=t + result.fit_results["t0"]) for t in result.t_list
    ]

    # Don't display the zero-inflated error bars
    cases_err = np.array(result.cases_stdev)
    cases_err[result.observed_new_cases == 0] = 0
    death_err = deepcopy(result.deaths_stdev)
    death_err[result.observed_new_deaths == 0] = 0
    if result.hosp_stdev is not None:
        hosp_stdev = deepcopy(result.hosp_stdev)
        hosp_stdev[hosp_stdev > 1e5] = 0

    # PLOT DATA
    fig, ax = plt.subplots(figsize=(18, 12))

    plt.errorbar(
        data_dates,
        result.observed_new_cases,
        yerr=cases_err,
        marker="o",
        linestyle="",
        label="Observed Cases Per Day",
        color="steelblue",
        capsize=3,
        alpha=0.4,
        markersize=10,
    )

    plt.errorbar(
        data_dates,
        result.observed_new_deaths,
        yerr=death_err,
        marker="d",
        linestyle="",
        label="Observed Deaths Per Day",
        color="firebrick",
        capsize=3,
        alpha=0.4,
        markersize=10,
    )

    plt.errorbar(
        icu_dates,
        result.icu,
        yerr=None,
        marker="o",
        linestyle="",
        label="Observed Current ICU",
        color="goldenrod",
        capsize=3,
        alpha=0.4,
        markersize=10,
    )

    plt.plot(
        model_dates,
        result.mle_model.results["total_new_infections"],
        label="Estimated Total New Infections Per Day",
        linestyle="--",
        lw=4,
        color="steelblue",
    )
    plt.plot(
        model_dates,
        result.fit_results["test_fraction"] * result.mle_model.results["total_new_infections"],
        label="Estimated Tested New Infections Per Day",
        color="steelblue",
        lw=4,
    )

    plt.plot(
        model_dates,
        result.mle_model.results["total_deaths_per_day"],
        label="Model Deaths Per Day",
        color="firebrick",
        lw=4,
    )

    if result.hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
        new_hosp_observed = result.hospitalizations[1:] - result.hospitalizations[:-1]
        plt.errorbar(
            hosp_dates[1:],
            new_hosp_observed,
            yerr=hosp_stdev,
            marker="s",
            linestyle="",
            label="Observed New Hospitalizations Per Day",
            color="darkseagreen",
            capsize=3,
            alpha=1,
        )
        predicted_hosp = (
            result.mle_model.results["HGen_cumulative"]
            + result.mle_model.results["HICU_cumulative"]
        )
        predicted_hosp = predicted_hosp[1:] - predicted_hosp[:-1]
        plt.plot(
            model_dates[1:],
            result.fit_results["hosp_fraction"] * predicted_hosp,
            label="Estimated Total New Hospitalizations Per Day",
            linestyle="-.",
            lw=4,
            color="darkseagreen",
            markersize=10,
        )
    elif result.hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
        plt.errorbar(
            hosp_dates,
            result.hospitalizations,
            yerr=hosp_stdev,
            marker="s",
            linestyle="",
            label="Observed Total Current Hospitalizations",
            color="darkseagreen",
            capsize=3,
            alpha=0.5,
            markersize=10,
        )
        predicted_hosp = result.mle_model.results["HGen"] + result.mle_model.results["HICU"]
        plt.plot(
            model_dates,
            result.fit_results["hosp_fraction"] * predicted_hosp,
            label="Estimated Total Current Hospitalizations",
            linestyle="-.",
            lw=4,
            color="darkseagreen",
        )

    plt.plot(
        model_dates,
        result.fit_results["hosp_fraction"] * result.mle_model.results["HICU"],
        label="Estimated ICU Occupancy",
        linestyle=":",
        lw=6,
        color="goldenrod",
    )
    plt.plot(
        model_dates,
        result.fit_results["hosp_fraction"] * result.mle_model.results["HGen"],
        label="Estimated General Occupancy",
        linestyle=":",
        lw=4,
        color="black",
        alpha=0.4,
    )

    TRANSITION = timedelta(days=14)
    FORECAST_WINDOW = timedelta(days=60)
    YSCALE_MULTIPLIER = 2
    model_days_to_t_break = timedelta(days=result.fit_results["t_break"] + result.fit_results["t0"])

    start_date_p1 = result.ref_date + model_days_to_t_break
    stop_date_p1 = start_date_p1 + TRANSITION

    start_date_p2 = stop_date_p1 + timedelta(days=result.fit_results["t_delta_phases"])
    stop_date_p2 = start_date_p2 + TRANSITION

    for start, stop, label in [
        (start_date_p1, stop_date_p1, "Phase 1 Transition"),
        (start_date_p2, stop_date_p2, "Phase 2 Transition"),
    ]:
        plt.axvspan(start, stop, alpha=0.2, label=label)

    first_nonzero_date = data_dates[np.argmax(result.observed_new_cases > 0)]
    if result.SEIR_kwargs["beds_ICU"] > 0:
        plt.axhline(
            result.SEIR_kwargs["beds_ICU"], color="k", linestyle="-", linewidth=6, alpha=0.2
        )
        plt.text(
            first_nonzero_date + timedelta(days=5),
            result.SEIR_kwargs["beds_ICU"] * 1.1,
            "Available ICU Capacity",
            color="k",
            alpha=0.5,
            fontsize=15,
        )

    ax.set_ylim(bottom=1, top=YSCALE_MULTIPLIER * max(result.observed_new_cases))
    ax.set_xlim(left=first_nonzero_date, right=data_dates[-1] + FORECAST_WINDOW)
    plt.xticks(rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend(loc="best", fontsize=14)
    plt.grid(which="both", alpha=0.5)
    plt.title(result.display_name, fontsize=60)

    def fmt_txt(value):
        if np.isscalar(value) and not isinstance(value, str):
            return f"{value:1.2f}"
        else:
            return f"{value}"

    TXT_OFFSET = 0.02
    LINE_HEIGHT = 0.032
    SUMMARY_START = 0.95
    DEBUG_START = 0.65
    SHARED_KWARGS = dict(x=1 + TXT_OFFSET, transform=plt.gca().transAxes, fontsize=15, alpha=0.6,)

    # Plot Headline Results. We want these ordered.
    summary_fields = ("chi2_total", "R0", "Reff", "Reff2", "eps", "eps2", "t_delta_phases", "fips")
    for i, key in enumerate(summary_fields):
        value = result.fit_results[key]
        plt.text(
            y=SUMMARY_START - LINE_HEIGHT * i,
            s=f"{key}={fmt_txt(value)}",
            fontweight="bold",
            **SHARED_KWARGS,
        )

    # Plot Debug Results. These can be unordered.
    i = 0
    for key, value in result.fit_results.items():
        if key not in summary_fields:
            plt.text(y=DEBUG_START - LINE_HEIGHT * i, s=f"{key}={fmt_txt(value)}", **SHARED_KWARGS)
            i += 1

    return plt.gcf()
