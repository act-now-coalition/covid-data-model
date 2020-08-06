import datetime
import logging
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import us
import pickle
import json
import copy
from collections import defaultdict
from pyseir.models.seir_model import SEIRModel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
import pyseir.models.suppression_policies as sp
from pyseir import load_data
from pyseir.utils import get_run_artifact_path, RunArtifact, RunMode
from pyseir.inference import fit_results
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets


_logger = logging.getLogger(__name__)


compartment_to_capacity_attr_map = {
    "HGen": "beds_general",
    "HICU": "beds_ICU",
    "HVent": "ventilators",
}


class EnsembleRunner:
    """
    The EnsembleRunner executes a collection of N_samples simulations based on
    priors defined in the ParameterEnsembleGenerator.

    Parameters
    ----------
    fips: str
        County or state fips code
    n_years: int
        Number of years to simulate
    n_samples: int
        Ensemble size to run for each suppression policy.
    suppression_policy: list(float or str)
        List of suppression policies to apply.
    output_percentiles: list
        List of output percentiles desired. These will be computed for each
        compartment.
    run_mode: str
        Individual parameters can be overridden here.
    min_hospitalization_threshold: int
        Require this number of hospitalizations before initializing based on
        observations. Fallback to cases otherwise.
    hospitalization_to_confirmed_case_ratio: float
        When hospitalization data is not available directly, this fraction of
        confirmed cases defines the initial number of hospitalizations.
    """

    def __init__(
        self,
        fips,
        n_years=0.5,
        n_samples=250,
        suppression_policy=(0.35, 0.5, 0.75, 1),
        skip_plots=False,
        output_percentiles=(5, 25, 32, 50, 75, 68, 95),
        run_mode=RunMode.DEFAULT,
        min_hospitalization_threshold=5,
        hospitalization_to_confirmed_case_ratio=1 / 4,
    ):

        self.fips = fips
        self.agg_level = AggregationLevel.COUNTY if len(fips) == 5 else AggregationLevel.STATE

        self.t_list = np.linspace(0, int(365 * n_years), int(365 * n_years) + 1)
        self.skip_plots = skip_plots
        self.run_mode = RunMode(run_mode)
        self.hospitalizations_for_state = None
        self.min_hospitalization_threshold = min_hospitalization_threshold
        self.hospitalization_to_confirmed_case_ratio = hospitalization_to_confirmed_case_ratio

        if self.agg_level is AggregationLevel.COUNTY:
            self.county_metadata = load_data.load_county_metadata_by_fips(fips)
            self.state_name = us.states.lookup(self.county_metadata["state"]).name
            self.output_file_data = get_run_artifact_path(self.fips, RunArtifact.ENSEMBLE_RESULT)
        else:
            self.state_name = us.states.lookup(self.fips).name
            self.output_file_data = get_run_artifact_path(self.fips, RunArtifact.ENSEMBLE_RESULT)

        os.makedirs(os.path.dirname(self.output_file_data), exist_ok=True)
        self.output_percentiles = output_percentiles
        self.n_samples = n_samples
        self.n_years = n_years
        self.date_generated = datetime.datetime.utcnow().isoformat()
        self.suppression_policy = suppression_policy
        self.summary = copy.deepcopy(self.__dict__)
        self.summary.pop("t_list")

        self.suppression_policies = None
        self.override_params = dict()
        self.init_run_mode()

        self.all_outputs = {}

    def init_run_mode(self):
        """
        Based on the run mode, generate suppression policies and ensemble
        parameters.  This enables different model combinations and project
        phases.
        """
        self.suppression_policies = dict()

        if self.run_mode is RunMode.CAN_INFERENCE_DERIVED:
            self.n_samples = 1
            for scenario in [
                "no_intervention",
                "flatten_the_curve",
                "inferred",
                "social_distancing",
            ]:
                self.suppression_policies[f"suppression_policy__{scenario}"] = scenario

        elif self.run_mode is RunMode.DEFAULT:
            for suppression_policy in self.suppression_policy:
                self.suppression_policies[
                    f"suppression_policy__{suppression_policy}"
                ] = sp.generate_empirical_distancing_policy(
                    t_list=self.t_list, fips=self.fips, future_suppression=suppression_policy
                )
            self.override_params = dict()
        else:
            raise ValueError("Invalid run mode.")

    @staticmethod
    def _run_single_simulation(parameter_set):
        """
        Run a single simulation instance.

        Parameters
        ----------
        parameter_set: dict
            Params passed to the SEIR model

        Returns
        -------
        model: SEIRModel
            Executed model.
        """
        model = SEIRModel(**parameter_set)
        model.run()
        return model

    def _load_model_for_fips(self, scenario="inferred"):
        """
        Try to load a model for the locale, else load the state level model
        and update parameters for the county.
        """
        artifact_path = get_run_artifact_path(self.fips, RunArtifact.MLE_FIT_MODEL)
        if os.path.exists(artifact_path):
            with open(artifact_path, "rb") as f:
                model = pickle.load(f)
            inferred_params = fit_results.load_inference_result(self.fips)

        else:
            _logger.info(
                f"No MLE model found for {self.state_name}: {self.fips}. Reverting to state level."
            )
            artifact_path = get_run_artifact_path(self.fips[:2], RunArtifact.MLE_FIT_MODEL)
            if os.path.exists(artifact_path):
                with open(artifact_path, "rb") as f:
                    model = pickle.load(f)
                inferred_params = fit_results.load_inference_result(self.fips[:2])
            else:
                raise FileNotFoundError(f"Could not locate state result for {self.state_name}")

            # Rescale state values to the county population and replace county
            # specific params.
            # TODO: get_average_seir_parameters should return the analytic solution when available
            # right now it runs an average over the ensemble (with N_samples not consistently set
            # across the code base).
            default_params = ParameterEnsembleGenerator(
                self.fips,
                N_samples=500,
                t_list=model.t_list,
                suppression_policy=model.suppression_policy,
            ).get_average_seir_parameters()
            population_ratio = default_params["N"] / model.N
            model.N *= population_ratio
            model.I_initial *= population_ratio
            model.E_initial *= population_ratio
            model.A_initial *= population_ratio
            model.S_initial = model.N - model.I_initial - model.E_initial - model.A_initial

            for key in {"beds_general", "beds_ICU", "ventilators"}:
                setattr(model, key, default_params[key])

        # Determine the appropriate future suppression policy based on the
        # scenario of interest.

        eps_final = sp.estimate_future_suppression_from_fits(inferred_params, scenario=scenario)

        model.suppression_policy = sp.get_epsilon_interpolator(
            eps=inferred_params["eps"],
            t_break=inferred_params["t_break"],
            eps2=inferred_params["eps2"],
            t_delta_phases=inferred_params["t_delta_phases"],
            t_break_final=(
                datetime.datetime.today()
                - datetime.datetime.fromisoformat(inferred_params["t0_date"])
            ).days,
            eps_final=eps_final,
        )
        model.run()
        return model

    def run_ensemble(self):
        """
        Run an ensemble of models for each suppression policy nad generate the
        output report / results dataset.
        """
        for suppression_policy_name, suppression_policy in self.suppression_policies.items():

            _logger.info(
                f"Running simulation ensemble for {self.state_name} {self.fips} {suppression_policy_name}"
            )

            if self.run_mode is RunMode.CAN_INFERENCE_DERIVED:
                model_ensemble = [self._load_model_for_fips(scenario=suppression_policy)]

            else:
                raise ValueError(f"Run mode {self.run_mode.value} not supported.")

            if self.agg_level is AggregationLevel.COUNTY:
                self.all_outputs["county_metadata"] = self.county_metadata
                self.all_outputs["county_metadata"]["age_distribution"] = list(
                    self.all_outputs["county_metadata"]["age_distribution"]
                )
                self.all_outputs["county_metadata"]["age_bins"] = list(
                    self.all_outputs["county_metadata"]["age_distribution"]
                )

            self.all_outputs[
                f"{suppression_policy_name}"
            ] = self._generate_output_for_suppression_policy(model_ensemble)

        with open(self.output_file_data, "w") as f:
            json.dump(self.all_outputs, f)

    @staticmethod
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

    @staticmethod
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
                m.results[compartment][::-1]
                > getattr(m, compartment_to_capacity_attr_map[compartment])
            )
            surge_end.append(
                m.t_list[::-1][surge_end_idx[0][0]] if len(surge_end_idx) > 0 else float("NaN")
            )

        return surge_start, surge_end

    def _detect_peak_time_and_value(self, value_stack, t_list):
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
        for percentile in self.output_percentiles:
            peak_data["peak_value_ci%i" % percentile] = np.percentile(
                values_at_peak_index, percentile
            ).tolist()
            peak_data["peak_time_ci%i" % percentile] = np.percentile(
                peak_times, percentile
            ).tolist()

        peak_data["peak_value_mean"] = np.mean(values_at_peak_index).tolist()
        return peak_data

    def _generate_output_for_suppression_policy(self, model_ensemble):
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
        for compartment, value_stack in self._generate_compartment_arrays(model_ensemble).items():
            compartment_output = dict()

            # Compute percentiles over the ensemble
            for percentile in self.output_percentiles:
                outputs[compartment]["ci_%i" % percentile] = np.percentile(
                    value_stack, percentile, axis=0
                ).tolist()

            if compartment in compartment_to_capacity_attr_map:
                (
                    compartment_output["surge_start"],
                    compartment_output["surge_start"],
                ) = self._get_surge_window(model_ensemble, compartment)
                compartment_output["capacity"] = [
                    getattr(m, compartment_to_capacity_attr_map[compartment])
                    for m in model_ensemble
                ]

            compartment_output.update(
                self._detect_peak_time_and_value(value_stack, outputs["t_list"])
            )

            # Merge this dictionary into the suppression level one.
            outputs[compartment].update(compartment_output)

        return outputs


def _run_county(fips, ensemble_kwargs):
    """
    Execute the ensemble runner for a specific county.

    Parameters
    ----------
    fips: str
        County fips.
    ensemble_kwargs: dict
        Kwargs passed to the EnsembleRunner object.
    """
    runner = EnsembleRunner(fips=fips, **ensemble_kwargs)
    runner.run_ensemble()


def run_state(state_full_name, ensemble_kwargs, states_only=False):
    """
    Run the EnsembleRunner for each county in a state.

    Parameters
    ----------
    state: str
        State to run against.
    ensemble_kwargs: dict
        Kwargs passed to the EnsembleRunner object.
    states_only: bool
        If True only run the state level.
    """
    # Run the state level
    state_obj = us.states.lookup(state_full_name)
    fips = state_obj.fips
    runner = EnsembleRunner(fips=fips, **ensemble_kwargs)
    runner.run_ensemble()

    if not states_only:
        # Run county level
        state = state_obj.abbr
        county_latest = combined_datasets.load_us_latest_dataset().county
        all_fips = county_latest.get_subset(state=state).all_fips
        with Pool(maxtasksperchild=1) as p:
            f = partial(_run_county, ensemble_kwargs=ensemble_kwargs)
            p.map(f, all_fips)
