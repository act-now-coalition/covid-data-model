from datetime import timedelta
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from pyseir.load_data import HospitalizationDataType  # Do we still need this?
from pyseir.utils import get_run_artifact_path, RunArtifact


def plot_fitting_results(result) -> None:
    """
    Take a model_fitter.ModuleFitter instance and produce the plots. 
    """
    data_dates = [result.ref_date + timedelta(days=t) for t in result.times]
    if result.hospital_times is not None:
        hosp_dates = [result.ref_date + timedelta(days=float(t)) for t in result.hospital_times]
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

    plt.figure(figsize=(18, 12))
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
        color="black",
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

    plt.yscale("log")
    y_lim = plt.ylim(0.8e0)

    start_intervention_date = result.ref_date + timedelta(
        days=result.fit_results["t_break"] + result.fit_results["t0"]
    )
    stop_intervention_date = start_intervention_date + timedelta(days=14)

    plt.fill_betweenx(
        [y_lim[0], y_lim[1]],
        [start_intervention_date, start_intervention_date],
        [stop_intervention_date, stop_intervention_date],
        alpha=0.2,
        label="Estimated Intervention",
    )

    start_intervention2_date = (
        result.ref_date
        + timedelta(
            days=result.fit_results["t_break"]
            + result.fit_results["t_delta_phases"]
            + result.fit_results["t0"]
        )
        + timedelta(days=14)
    )
    stop_intervention2_date = start_intervention2_date + timedelta(days=14)

    plt.fill_betweenx(
        [y_lim[0], y_lim[1]],
        [start_intervention2_date, start_intervention2_date],
        [stop_intervention2_date, stop_intervention2_date],
        alpha=0.2,
        label="Estimated Intervention2",
    )

    running_total = timedelta(days=0)
    for i_label, k in enumerate(
        (
            "symptoms_to_hospital_days",
            "hospitalization_length_of_stay_general",
            "hospitalization_length_of_stay_icu",
        )
    ):
        end_time = timedelta(days=result.SEIR_kwargs[k])
        x = start_intervention_date + running_total
        y = 1.5 ** (i_label + 1)
        plt.errorbar(
            x=[x],
            y=[y],
            xerr=[[timedelta(days=0)], [end_time]],
            marker="",
            capsize=8,
            color="k",
            elinewidth=3,
            capthick=3,
        )
        plt.text(x + (end_time + timedelta(days=2)), y, k.replace("_", " ").title(), fontsize=14)
        running_total += end_time

    if result.SEIR_kwargs["beds_ICU"] > 0:
        plt.hlines(
            result.SEIR_kwargs["beds_ICU"],
            *plt.xlim(),
            color="k",
            linestyles="-",
            linewidths=6,
            alpha=0.2,
        )
        plt.text(
            data_dates[0] + timedelta(days=5),
            result.SEIR_kwargs["beds_ICU"] * 1.1,
            "Available ICU Capacity",
            color="k",
            alpha=0.5,
            fontsize=15,
        )

    plt.ylim(*y_lim)
    plt.xlim(min(model_dates[0], data_dates[0]), data_dates[-1] + timedelta(days=150))
    plt.xticks(rotation=30, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc=4, fontsize=14)
    plt.grid(which="both", alpha=0.5)
    plt.title(result.display_name, fontsize=60)

    chi_total = 0
    for i, (k, v) in enumerate(result.fit_results.items()):
        if k in ("chi2_cases", "chi2_deaths", "chi2_hosps"):
            chi_total += v

    for i, (k, v) in enumerate(result.fit_results.items()):

        fontweight = (
            "bold" if k in ("R0", "Reff", "Reff2", "eps", "eps2", "t_delta_phases") else "normal"
        )

        if np.isscalar(v) and not isinstance(v, str):
            plt.text(
                1.05,
                0.7 - 0.032 * i,
                f"{k}={v:1.3f}",
                transform=plt.gca().transAxes,
                fontsize=15,
                alpha=0.6,
                fontweight=fontweight,
            )

        else:
            plt.text(
                1.05,
                0.7 - 0.032 * i,
                f"{k}={v}",
                transform=plt.gca().transAxes,
                fontsize=15,
                alpha=0.6,
                fontweight=fontweight,
            )
    plt.text(
        1.05,
        0.75,
        f"total_chi2:{chi_total:1.3f}",
        transform=plt.gca().transAxes,
        fontsize=15,
        alpha=0.6,
        fontweight="bold",
    )
    output_file = get_run_artifact_path(result.fips, RunArtifact.MLE_FIT_REPORT)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    result.mle_model.plot_results()
    plt.savefig(output_file.replace("mle_fit_results", "mle_fit_model"), bbox_inches="tight")
    plt.close()
