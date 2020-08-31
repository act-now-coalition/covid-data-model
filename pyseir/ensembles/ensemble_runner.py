import datetime
import os
from dataclasses import dataclass
from typing import Mapping, Any, Optional

import numpy as np

import structlog
import json
import copy
from collections import defaultdict

from libs import pipeline
from pyseir.inference import model_fitter
from pyseir.models import seir_model
from pyseir.models.seir_model import SEIRModel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
import pyseir.models.suppression_policies as sp
from pyseir.utils import RunArtifact


_log = structlog.get_logger()


compartment_to_capacity_attr_map = {
    "HGen": "beds_general",
    "HICU": "beds_ICU",
    "HVent": "ventilators",
}


@dataclass(frozen=True)
class RegionalInput:
    region: pipeline.Region

    _combined_data: pipeline.RegionalCombinedData
    _mle_fit_model: seir_model.SEIRModel
    _mle_fit_result: Mapping[str, Any]
    _state_mle_fit_model: Optional[seir_model.SEIRModel] = None
    _state_mle_fit_result: Optional[Mapping[str, Any]] = None

    @staticmethod
    def for_state(fitter: model_fitter.ModelFitter) -> "RegionalInput":
        return RegionalInput(
            region=fitter.region,
            _combined_data=fitter.regional_input._combined_data,
            _mle_fit_model=fitter.mle_model,
            _mle_fit_result=fitter.fit_results,
        )

    @staticmethod
    def for_substate(
        fitter: model_fitter.ModelFitter, state_fitter: model_fitter.ModelFitter
    ) -> "RegionalInput":
        return RegionalInput(
            region=fitter.region,
            _combined_data=fitter.regional_input._combined_data,
            _mle_fit_model=fitter.mle_model,
            _mle_fit_result=fitter.fit_results,
            _state_mle_fit_model=state_fitter.mle_model,
            _state_mle_fit_result=state_fitter.fit_results,
        )

    @property
    def fips(self) -> str:
        return self.region.fips

    def state_name(self):
        return self.region.state_obj().name

    def get_us_latest(self) -> Mapping[str, Any]:
        return self._combined_data.get_us_latest()

    def load_mle_fit_model(self) -> Optional[seir_model.SEIRModel]:
        return self._mle_fit_model

    def load_inference_result(self):
        return self._mle_fit_result

    def load_state_mle_fit_model(self) -> Optional[seir_model.SEIRModel]:
        return self._state_mle_fit_model

    def load_state_inference_result(self):
        return self._state_mle_fit_result


class EnsembleRunner:
    """
    The EnsembleRunner executes a collection of N_samples simulations based on
    priors defined in the ParameterEnsembleGenerator.

    Parameters
    ----------
    regional_input: RegionalInput
        County or state data
    n_years: int
        Number of years to simulate
    n_samples: int
        Ensemble size to run for each suppression policy.
    suppression_policy: list(float or str)
        List of suppression policies to apply.
    output_percentiles: list
        List of output percentiles desired. These will be computed for each
        compartment.
    min_hospitalization_threshold: int
        Require this number of hospitalizations before initializing based on
        observations. Fallback to cases otherwise.
    hospitalization_to_confirmed_case_ratio: float
        When hospitalization data is not available directly, this fraction of
        confirmed cases defines the initial number of hospitalizations.
    """

    def __init__(
        self,
        regional_input: RegionalInput,
        n_years=0.5,
        n_samples=250,
        suppression_policy=(0.35, 0.5, 0.75, 1),
        skip_plots=False,
        output_percentiles=(5, 25, 32, 50, 75, 68, 95),
        min_hospitalization_threshold=5,
        hospitalization_to_confirmed_case_ratio=1 / 4,
    ):

        self.regional_input = regional_input

        self.t_list = np.linspace(0, int(365 * n_years), int(365 * n_years) + 1)
        self.skip_plots = skip_plots
        self.hospitalizations_for_state = None
        self.min_hospitalization_threshold = min_hospitalization_threshold
        self.hospitalization_to_confirmed_case_ratio = hospitalization_to_confirmed_case_ratio

        self.state_name = regional_input.state_name()
        self.output_file_data = self.regional_input.region.run_artifact_path_to_write(
            RunArtifact.ENSEMBLE_RESULT
        )

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
        self.initialize_suppression_policies()

        self.all_outputs = {}

    def initialize_suppression_policies(self):
        """
        Based on the run mode, generate suppression policies and ensemble
        parameters.  This enables different model combinations and project
        phases.
        """
        self.suppression_policies = dict()

        self.n_samples = 1
        for scenario in [
            "no_intervention",
            "flatten_the_curve",
            "inferred",
            "social_distancing",
        ]:
            self.suppression_policies[f"suppression_policy__{scenario}"] = scenario

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

    def _load_model_for_region(self, scenario="inferred"):
        """
        Try to load a model for the region, else load the state level model and update parameters
        for the region.
        """
        model = self.regional_input.load_mle_fit_model()
        if model:
            inferred_params = self.regional_input.load_inference_result()
        else:
            _log.info(
                f"No MLE model found. Reverting to state level.", region=self.regional_input.region
            )
            model = self.regional_input.load_state_mle_fit_model()
            if model:
                inferred_params = self.regional_input.load_state_inference_result()
            else:
                raise FileNotFoundError(f"Could not locate state result for {self.state_name}")

            # Rescale state values to the county population and replace county
            # specific params.
            # TODO: get_average_seir_parameters should return the analytic solution when available
            # right now it runs an average over the ensemble (with N_samples not consistently set
            # across the code base).
            default_params = ParameterEnsembleGenerator(
                N_samples=500,
                combined_datasets_latest=self.regional_input.get_us_latest(),
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

            _log.info(
                "Running simulation ensemble",
                suppression_policy=suppression_policy_name,
                region=self.regional_input.region,
            )

            model_ensemble = [self._load_model_for_region(scenario=suppression_policy)]

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
            key: [] for key in model_ensemble[0].results.keys() if key not in ("t_list")
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


def run_region(regional_input: RegionalInput):
    """
    Run the EnsembleRunner for each county in a state.

    Parameters
    ----------
    regional_input: RegionalInput
        Region to run against.
    """
    # Run the state level
    runner = EnsembleRunner(regional_input=regional_input)
    runner.run_ensemble()
