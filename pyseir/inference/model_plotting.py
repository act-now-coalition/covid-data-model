from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

from pyseir.load_data import HospitalizationDataType  # Do we still need this?
from pyseir.utils import get_run_artifact_path, RunArtifact


def plot_fitting_results(result: "pyseir.inference.ModelFitter") -> None:
    """
    Entry point from model_fitter. Generate and save all PySEIR related Figures.
    """
    output_file = result.region.run_artifact_path_to_write(RunArtifact.MLE_FIT_REPORT)

    # Save the mle fitter
    mle_fig = result.mle_model.plot_results()
    mle_fig.savefig(output_file.replace("mle_fit_results", "mle_fit_model"), bbox_inches="tight")

    # Generate Figure
    fig = _plot_model_fitter_results(result)

    # Save the figure in Log-Linear
    fig.gca().set_yscale("log")
    fig.savefig(output_file, bbox_inches="tight")

    # Save the figure in Linear mode
    fig.gca().set_yscale("linear")
    fig.savefig(
        output_file.replace("mle_fit_results", "mle_fit_results_linear"), bbox_inches="tight"
    )
    return


def _plot_model_fitter_results(result) -> plt.Figure:
    """
    Take a model_fitter.ModelFitter instance and produce the plots.
    """
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot Desired Timeseries from Observed Data
    for ts in _get_observed_timeseries_to_plot(result):
        x = [result.ref_date + timedelta(days=float(t)) for t in ts["times"]]
        y = ts["y"]
        error = np.where(y == 0, 0, ts["error"])
        ax.errorbar(
            x=x,
            y=y,
            yerr=error,
            marker=ts["marker"],
            label=ts["label"],
            color=ts["color"],
            linestyle="",
            capsize=3,
            alpha=0.4,
            markersize=10,
        )

    # Plot Desired Timeseries from Observed Data
    for ts in _get_model_timeseries_to_plot(result):
        ax.plot(
            ts["x"],
            ts["y"],
            label=ts["label"],
            linestyle=ts["linestyle"],
            color=ts["color"],
            lw=ts["lw"],
        )

    # Plot the transition graphs.
    fig = _add_annotations(result=result, figure=fig)

    # Add table of Fit Results
    fig = _add_table(result=result, figure=fig, debug=True)

    # Set Figure Level Details
    YSCALE_MULTIPLIER = 2  # Set ylim upper to a multiple of the max observed_new_cases
    FORECAST_WINDOW = timedelta(days=60)
    data_dates = [result.ref_date + timedelta(days=t) for t in result.times]
    first_nonzero_date = data_dates[np.argmax(result.observed_new_cases > 0)]  # First nonzero date

    ax.set_ylim(bottom=1, top=YSCALE_MULTIPLIER * max(result.observed_new_cases))
    ax.set_xlim(left=first_nonzero_date, right=data_dates[-1] + FORECAST_WINDOW)
    plt.xticks(rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend(loc="best", fontsize=14)
    plt.grid(which="both", alpha=0.5)
    plt.title(result.display_name, fontsize=60)

    return fig


def _get_observed_timeseries_to_plot(result) -> list:
    """
    Generate a list of dictionaries where each dictionary is the params for the errorbar plotting.
    """
    output = list()
    # Load and Plot Cases
    output.append(
        dict(  # Case Data
            times=result.times,
            y=result.observed_new_cases,
            error=np.array(result.cases_stdev),
            label="Observed Cases Per Day",
            color="steelblue",
            marker="o",
        )
    )
    output.append(
        dict(  # Death Data
            times=result.times,
            y=result.observed_new_deaths,
            error=np.array(result.deaths_stdev),
            label="Observed Deaths Per Day",
            color="firebrick",
            marker="d",
        )
    )

    if result.icu_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
        output.append(
            dict(
                times=result.icu_times,
                y=result.icu,
                error=np.zeros_like(result.icu),
                label="Observed Current ICU",
                color="goldenrod",
                marker="o",
            )
        )
    elif result.icu_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
        pass  # Don't plot this converted type.

    # If Current
    if result.hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
        output.append(
            dict(
                times=result.hospital_times,
                y=result.hospitalizations,
                error=np.where(result.hosp_stdev > 1e5, 0, result.hosp_stdev),
                # This error correction is in the original code. Brett doesn't know why.
                label="Observed Total Current Hospitalization",
                color="darkseagreen",
                marker="s",
            )
        )
    elif result.hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
        # Only still plotting this because we score on this in the SEIR model
        output.append(
            dict(
                times=result.hospital_times[1:],
                y=result.hospitalizations[1:] - result.hospitalizations[:-1],
                error=np.where(result.hosp_stdev > 1e5, 0, result.hosp_stdev),
                label="Observed Incident Hospitalization",
                color="darkseagreen",
                marker="s",
            )
        )
    return output


def _get_model_timeseries_to_plot(result) -> list:
    """
    """
    model_dates = [
        result.ref_date + timedelta(days=t + result.fit_results["t0"]) for t in result.t_list
    ]
    output = [
        dict(
            x=model_dates,
            y=result.mle_model.results["total_new_infections"],
            label="Estimated Total New Infections Per Day",
            linestyle="--",
            color="steelblue",
            lw=4,
        ),
        dict(
            x=model_dates,
            y=(
                result.fit_results["test_fraction"]
                * result.mle_model.results["total_new_infections"]
            ),
            label="Estimated Tested New Infections Per Day",
            linestyle="-",
            color="steelblue",
            lw=4,
        ),
        dict(
            x=model_dates,
            y=result.mle_model.results["total_deaths_per_day"],
            label="Model Deaths Per Day",
            linestyle="-",
            color="firebrick",
            lw=4,
        ),
        dict(
            x=model_dates,
            y=result.fit_results["hosp_fraction"] * result.mle_model.results["HICU"],
            label="Estimated ICU Occupancy",
            linestyle=":",
            color="goldenrod",
            lw=6,
        ),
        dict(
            x=model_dates,
            y=result.fit_results["hosp_fraction"] * result.mle_model.results["HGen"],
            label="Estimated Generaly Occupancy",
            linestyle=":",
            color="grey",
            lw=4,
        ),
    ]

    if result.hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
        predicted_hosp = (
            result.mle_model.results["HGen_cumulative"]
            + result.mle_model.results["HICU_cumulative"]
        )
        predicted_hosp = predicted_hosp[1:] - predicted_hosp[:-1]
        output.append(
            dict(
                x=model_dates[1:],
                y=result.fit_results["hosp_fraction"] * predicted_hosp,
                label="Estimated Total New Hospitalizations Per Day",
                linestyle="-.",
                lw=4,
                color="darkseagreen",
            )
        )
    elif result.hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
        predicted_hosp = result.mle_model.results["HGen"] + result.mle_model.results["HICU"]
        output.append(
            dict(
                x=model_dates,
                y=result.fit_results["hosp_fraction"] * predicted_hosp,
                label="Estimated Total Current Hospitalizations",
                linestyle="-.",
                lw=4,
                color="darkseagreen",
            )
        )

    return output


def _add_annotations(result, figure) -> plt.Figure:
    """"""
    TRANSITION = timedelta(days=14)
    model_days_to_t_break = timedelta(days=result.fit_results["t_break"] + result.fit_results["t0"])

    start_date_p1 = result.ref_date + model_days_to_t_break
    stop_date_p1 = start_date_p1 + TRANSITION

    start_date_p2 = stop_date_p1 + timedelta(days=result.fit_results["t_delta_phases"])
    stop_date_p2 = start_date_p2 + TRANSITION

    for start, stop, label in [
        (start_date_p1, stop_date_p1, "Phase 1 Transition"),
        (start_date_p2, stop_date_p2, "Phase 2 Transition"),
    ]:
        figure.gca().axvspan(start, stop, alpha=0.2, label=label)

    # Label the Total ICU Available
    data_dates = [result.ref_date + timedelta(days=t) for t in result.times]
    first_nonzero_date = data_dates[np.argmax(result.observed_new_cases > 0)]
    if result.SEIR_kwargs["beds_ICU"] > 0:
        figure.gca().axhline(
            result.SEIR_kwargs["beds_ICU"], color="k", linestyle="-", linewidth=6, alpha=0.2
        )
        figure.gca().text(
            first_nonzero_date + timedelta(days=5),
            result.SEIR_kwargs["beds_ICU"] * 1.1,
            "Available ICU Capacity",
            color="k",
            alpha=0.5,
            fontsize=15,
        )

    return figure


def _add_table(result, figure, debug):
    """"""

    def fmt_txt(s):
        if np.isscalar(s) and not isinstance(s, str):
            return f"{s:1.2f}"
        else:
            return f"{s}"

    TXT_OFFSET = 0.02
    LINE_HEIGHT = 0.032
    SUMMARY_START = 0.95
    DEBUG_START = 0.65
    SHARED_KWARGS = dict(x=1 + TXT_OFFSET, transform=plt.gca().transAxes, fontsize=15, alpha=0.6,)

    # Plot Headline Results. We want these ordered.
    summary_fields = ("chi2_total", "R0", "Reff", "Reff2", "eps", "eps2", "t_delta_phases", "fips")
    for i, key in enumerate(summary_fields):
        value = result.fit_results[key]
        figure.gca().text(
            y=SUMMARY_START - LINE_HEIGHT * i,
            s=f"{key}={fmt_txt(value)}",
            fontweight="bold",
            **SHARED_KWARGS,
        )
    if debug:
        # Plot Debug Results. These can be unordered.
        i = 0
        for key, value in result.fit_results.items():
            if key not in summary_fields:
                figure.gca().text(
                    y=DEBUG_START - LINE_HEIGHT * i, s=f"{key}={fmt_txt(value)}", **SHARED_KWARGS
                )
                i += 1

    return figure
