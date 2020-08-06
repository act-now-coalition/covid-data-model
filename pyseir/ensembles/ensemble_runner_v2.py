import datetime
import numpy as np
from collections import defaultdict

from pyseir.inference import model_fitter


def generate_ensemble(inferred_params: dict) -> dict:
    """

    Take the inputs of an trained MLE model and return an ensemble


    Note: No longer supporting different RunModes -> Only using CAN Inference Derived
    """
    results = dict()
    for policy, r_forecasted in [
        ("no_intervention", 2.5),
        ("flatten_the_curve", 0.97),
        ("inferred", inferred_params["R0"]),
        ("social_distancing", 1.7),
    ]:

        model = model_fitter.ModelFitter(fips=inferred_params["fips"])
        output = model.run_model(
            R0=inferred_params["R0"],
            eps=inferred_params["eps"],
            t_break=inferred_params["t_break"],
            eps2=inferred_params["eps2"],
            t_delta_phases=inferred_params["t_delta_phases"],
            log10_I_initial=inferred_params["log10_I_initial"],
            t_break_final=(
                datetime.datetime.today()
                - datetime.datetime.fromisoformat(inferred_params["t0_date"])
            ).days,
            eps_final=r_forecasted / inferred_params["R0"],
        )
        results[f"suppression_policy__{policy}"] = _generate_output_for_suppression_policy([output])

    return results


# REMOVE ALL THIS BELOW FOR ROUND 2. It's just convoluted structure.##


compartment_to_capacity_attr_map = {
    "HGen": "beds_general",
    "HICU": "beds_ICU",
    "HVent": "ventilators",
}


def _generate_compartment_arrays(model_ensemble):
    """
    Given a collection of SEIR models, convert these to numpy arrays for
    each compartment, with axis 0 being the model index and axis 1 being the
    timestep.

    Parameters
    ----------
    model_ensemble: list(SEIRModel)

    Returns
    -------
    value_stack: array[n_samples, time steps]
        Array with the stacked model output results.
    """
    compartments = {
        key: []
        for key in model_ensemble[0].results.keys()
        if key not in ("t_list", "county_metadata")
    }
    for model in model_ensemble:
        for key in compartments:
            compartments[key].append(model.results[key])

    return {key: np.vstack(value_stack) for key, value_stack in compartments.items()}


def _get_surge_window(model_ensemble, compartment):
    """
    Calculate the list of surge window starts and ends for an ensemble.

    Parameters
    ----------
    model_ensemble: list(SEIRModel)
        List of models to compute the surge windows for.
    compartment: str
        Compartment to calculate the surge window over.

    Returns
    -------
    surge_start: np.array
        For each model, the surge start window time (since beginning of
        simulation). NaN implies no surge occurred.
    surge_end: np.array
        For each model, the surge end window time (since beginning of
        simulation). NaN implies no surge occurred.
    """
    surge_start = []
    surge_end = []
    for m in model_ensemble:
        # Find the first t where overcapacity occurs
        surge_start_idx = np.argwhere(
            m.results[compartment] > getattr(m, compartment_to_capacity_attr_map[compartment])
        )
        surge_start.append(
            m.t_list[surge_start_idx[0][0]] if len(surge_start_idx) > 0 else float("NaN")
        )

        # Reverse the t-list and capacity and do the same.
        surge_end_idx = np.argwhere(
            m.results[compartment][::-1] > getattr(m, compartment_to_capacity_attr_map[compartment])
        )
        surge_end.append(
            m.t_list[::-1][surge_end_idx[0][0]] if len(surge_end_idx) > 0 else float("NaN")
        )

    return surge_start, surge_end


def _detect_peak_time_and_value(value_stack, t_list):
    """
    Compute the peak times for each compartment by finding the arg
    max, and selecting the corresponding time.

    Parameters
    ----------
    value_stack: array[n_samples, time steps]
        Array with the stacked model output results.
    t_list: array
        Array of timesteps.

    Returns
    -------
    peak_data: dict
        For each confidence interval, produce key, value pairs for e.g.
            - peak_time_cl50
            - peak_value_cl50
        Also add peak_value_mean.
    """
    peak_indices = value_stack.argmax(axis=1)
    peak_times = [t_list[peak_index] for peak_index in peak_indices]
    values_at_peak_index = [val[idx] for val, idx in zip(value_stack, peak_indices)]

    peak_data = dict()
    output_percentiles = ((5, 25, 32, 50, 75, 68, 95),)
    for percentile in output_percentiles:
        peak_data[f"peak_value_ci{percentile}"] = np.percentile(
            values_at_peak_index, percentile
        ).tolist()
        peak_data["peak_time_ci{percentile}"] = np.percentile(peak_times, percentile).tolist()

    peak_data["peak_value_mean"] = np.mean(values_at_peak_index).tolist()
    return peak_data


def _generate_output_for_suppression_policy(model_ensemble):
    """
    Generate output data for a given suppression policy.

    Parameters
    ----------
    model_ensemble: list(SEIRModel)
        List of models to compute the surge windows for.

    Returns
    -------
    outputs: dict
        Output data for this suppression policc ensemble.
    """
    outputs = defaultdict(dict)
    outputs["t_list"] = model_ensemble[0].t_list.tolist()

    # ------------------------------------------
    # Calculate Confidence Intervals and Peaks
    # ------------------------------------------
    for compartment, value_stack in _generate_compartment_arrays(model_ensemble).items():
        compartment_output = dict()

        # Compute percentiles over the ensemble
        output_percentiles = ((5, 25, 32, 50, 75, 68, 95),)
        for percentile in output_percentiles:
            outputs[compartment][f"ci_{percentile}"] = np.percentile(
                value_stack, percentile, axis=0
            ).tolist()

        if compartment in compartment_to_capacity_attr_map:
            (
                compartment_output["surge_start"],
                compartment_output["surge_start"],
            ) = _get_surge_window(model_ensemble, compartment)
            compartment_output["capacity"] = [
                getattr(m, compartment_to_capacity_attr_map[compartment]) for m in model_ensemble
            ]

        compartment_output.update(_detect_peak_time_and_value(value_stack, outputs["t_list"]))

        # Merge this dictionary into the suppression level one.
        outputs[compartment].update(compartment_output)

    return outputs


if __name__ == "__main__":
    print("Running Main")
    example_inputs = {
        "fips": "16",
        "R0": 4.1005,
        "t0": 65.6485,
        "eps": 0.2013,
        "t_break": 12.5516,
        "eps2": 0.3516,
        "t_delta_phases": 44.2424,
        "test_fraction": 0.1664,
        "hosp_fraction": 0.9945,
        "log10_I_initial": 0.7625,
        "R0_error": 0.0198,
        "t0_error": 0.6788,
        "eps_error": 0.001,
        "t_break_error": 0.1855,
        "eps2_error": 0.0023,
        "t_delta_phases_error": 1.4806,
        "test_fraction_error": 0.0067,
        "hosp_fraction_error": 0.0076,
        "log10_I_initial_error": 0.0281,
        "t0_date": "2020-03-06T15:33:48.569238",
        "t_today": 212,
        "Reff": 0.8255,
        "Reff2": 1.4419,
        "chi2_cases": 107.7376,
        "chi2_hosps": 34.1433,
        "chi2_deaths": 9.9217,
        "chi2_total": 151.8026,
        "hospitalization_data_type": "current_hospitalizations",
    }
    print(generate_ensemble(example_inputs).keys())
