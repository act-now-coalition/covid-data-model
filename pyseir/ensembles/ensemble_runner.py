import datetime
import logging
import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import us
import json
from enum import Enum
import copy
from collections import defaultdict
from pyseir.models.seir_model import SEIRModel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from pyseir.models.suppression_policies import generate_empirical_distancing_policy, generate_covidactnow_scenarios
from pyseir import OUTPUT_DIR
from pyseir import load_data
from pyseir.reports.county_report import CountyReport
from libs.datasets import JHUDataset
from libs.datasets.dataset_utils import AggregationLevel

_logger = logging.getLogger(__name__)
jhu_timeseries = None


class RunMode(Enum):
    # Read params from the parameter sampler default and use empirical
    # suppression policies.
    DEFAULT = 'default'
    # 4 basic suppression scenarios and specialized parameters to match
    # covidactnow before scenarios.
    CAN_BEFORE = 'can-before'


compartment_to_capacity_attr_map = {
    'HGen': 'beds_general',
    'HICU': 'beds_ICU',
    'HVent': 'ventilators'
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
    """
    def __init__(self, fips, n_years=2, n_samples=250,
                 suppression_policy=(0.35, 0.5, 0.75, 1),
                 skip_plots=False,
                 output_percentiles=(5, 25, 32, 50, 75, 68, 95),
                 generate_report=True,
                 run_mode=RunMode.DEFAULT):

        # Caching globally to avoid relatively significant performance overhead
        # of loading for each county.
        global jhu_timeseries
        if not jhu_timeseries:
            jhu_timeseries = JHUDataset.local().timeseries()

        self.fips = fips
        self.geographic_unit = 'county' if len(self.fips) == 5 else 'state'

        self.t_list = np.linspace(0, 365 * n_years, 365 * n_years)
        self.skip_plots = skip_plots
        self.run_mode = RunMode(run_mode)

        if self.geographic_unit == 'county':
            self.county_metadata = load_data.load_county_metadata_by_fips(fips)
            self.state_abbr = us.states.lookup(self.county_metadata['state']).abbr
            self.state_name = us.states.lookup(self.county_metadata['state']).name

            self.output_file_report = os.path.join(OUTPUT_DIR, self.state_name, 'reports',
                f"{self.state_name}__{self.county_metadata['county']}__{self.fips}__{self.run_mode.value}__ensemble_projections.pdf")
            self.output_file_data = os.path.join(OUTPUT_DIR, self.state_name, 'data',
                f"{self.state_name}__{self.county_metadata['county']}__{self.fips}__{self.run_mode.value}__ensemble_projections.json")

            self.covid_data = jhu_timeseries.get_subset(AggregationLevel.COUNTY, country='USA', state=self.state_abbr)\
                                        .get_data(state=self.state_abbr, country='USA', fips=self.fips)
        else:
            self.state_abbr = us.states.lookup(self.fips).abbr
            self.state_name = us.states.lookup(self.fips).name
            self.covid_data = jhu_timeseries.get_subset(AggregationLevel.STATE, country='USA', state=self.state_abbr)\
                                        .get_data(country='USA', state=self.state_abbr)
            self.output_file_report = None
            self.output_file_data = os.path.join(OUTPUT_DIR, self.state_name, 'data',
                f"{self.state_name}__{self.fips}__{self.run_mode.value}__ensemble_projections.json")

        self.output_percentiles = output_percentiles
        self.n_samples = n_samples
        self.n_years = n_years
        # TODO: Will be soon replaced with loaders for all the inferred params.
        # self.t0 = fit_results.load_t0(fips)
        self.date_generated = datetime.datetime.utcnow().isoformat()
        self.suppression_policy = suppression_policy
        self.summary = copy.deepcopy(self.__dict__)
        self.summary.pop('t_list')
        self.generate_report = generate_report

        self.suppression_policies = None
        self.override_params = None
        self.init_run_mode()

        self.all_outputs = {}

    def init_run_mode(self):
        """
        Based on the run mode, generate suppression policies and ensemble
        parameters.  This enables different model combinations and project
        phases.
        """
        self.suppression_policies = dict()

        if self.run_mode is RunMode.CAN_BEFORE:
            self.n_samples = 1

            for scenario in ['no_intervention', 'flatten_the_curve', 'full_containment', 'social_distancing']:
                R0 = 3.6
                policy = generate_covidactnow_scenarios(t_list=self.t_list, R0=R0, t0=datetime.datetime.today(), scenario=scenario)
                self.suppression_policies[f'suppression_policy__{scenario}'] = policy
                self.override_params = ParameterEnsembleGenerator(
                    self.fips, N_samples=500, t_list=self.t_list, suppression_policy=policy).get_average_seir_parameters()

            self.override_params['R0'] = R0
            self.override_params['delta'] = 1 / 6.
            self.override_params['sigma'] = 1 / 3.

            self.override_params['mortality_rate'] = 0.0109
            self.override_params['mortality_rate_no_general_beds'] = 0.10
            # TODO: This is not modeled in CAN. Plan to increase.
            self.override_params['mortality_rate_no_ICU_beds'] = 0.00

            self.override_params['hospitalization_length_of_stay_general'] = 6
            self.override_params['hospitalization_length_of_stay_icu'] = 14
            self.override_params['hospitalization_length_of_stay_icu_and_ventilator'] = 14

            # hospitalization_rate_general is hospitalization rate for symptomatic cases going to the hospital
            # We ensure later that this adds up to a total overall 0.0727 rate.
            # self.override_params['hospitalization_rate_general'] = 0.25
            self.override_params['hospitalization_rate_general'] = 0.0727
            self.override_params['hospitalization_rate_icu'] = 0.1397 * self.override_params['hospitalization_rate_general']
            self.override_params['beds_ICU'] = 0 #.11 * self.override_params['beds_general'] # National average per hospital bed.
            self.override_params['symptoms_to_hospital_days'] = 6

            if len(self.covid_data) > 0 and self.covid_data.cases.max() > 0:
                self.t0 = self.covid_data.date.max()

                # Initial hospitalizations = Confirmed Cases / 4.
                # TODO Integrate actual hospital data here.
                hospitalization_to_confirmed_case_ratio = 1 / 4

                self.override_params['HGen_initial'] = self.covid_data.cases.max() * hospitalization_to_confirmed_case_ratio * (1 - self.override_params['hospitalization_rate_icu'])
                self.override_params['HICU_initial'] = self.covid_data.cases.max() * hospitalization_to_confirmed_case_ratio * self.override_params['hospitalization_rate_icu']
                self.override_params['HICUVent_initial'] = self.override_params['HICU_initial'] * self.override_params['fraction_icu_requiring_ventilator']

                self.override_params['I_initial'] = 1.0 * 3.43 * self.covid_data.cases.max()
                self.override_params['A_initial'] = 0
                self.override_params['gamma'] = 1

                self.override_params['E_initial'] = 1.2 * (self.override_params['I_initial'] + self.override_params['A_initial'])
                self.override_params['D_initial'] = self.covid_data.deaths.max()

            else:
                self.t0 = datetime.datetime.today()
                self.override_params['I_initial'] = 1
                self.override_params['A_initial'] = 1

        elif self.run_mode is RunMode.DEFAULT:
            for suppression_policy in self.suppression_policy:
                self.suppression_policies[f'suppression_policy__{suppression_policy}']= generate_empirical_distancing_policy(
                    t_list=self.t_list, fips=self.fips, future_suppression=suppression_policy)
            self.override_params = dict()
        else:
            raise ValueError('Invalid run mode.')

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

    def run_ensemble(self):
        """
        Run an ensemble of models for each suppression policy nad generate the
        output report / results dataset.
        """
        for suppression_policy_name, suppression_policy in self.suppression_policies.items():

            logging.info(f'Running simulation ensemble for {self.state_name} {self.fips} {suppression_policy_name}')

            parameter_sampler = ParameterEnsembleGenerator(
                fips=self.fips,
                N_samples=self.n_samples,
                t_list=self.t_list,
                suppression_policy=suppression_policy)
            parameter_ensemble = parameter_sampler.sample_seir_parameters(override_params=self.override_params)
            model_ensemble = list(map(self._run_single_simulation, parameter_ensemble))

            logging.info(f'Generating outputs for {suppression_policy_name}')
            if self.geographic_unit == 'county':
                self.all_outputs['county_metadata'] = self.county_metadata
                self.all_outputs['county_metadata']['age_distribution'] = list(self.all_outputs['county_metadata']['age_distribution'])
                self.all_outputs['county_metadata']['age_bins'] = list(self.all_outputs['county_metadata']['age_distribution'])

            self.all_outputs[f'{suppression_policy_name}'] = self._generate_output_for_suppression_policy(model_ensemble)

        if self.generate_report and self.output_file_report:
            logging.info(f'Generating report for {self.state_name} {self.fips}')
            report = CountyReport(self.fips,
                                  model_ensemble=model_ensemble,
                                  county_outputs=self.all_outputs,
                                  filename=self.output_file_report,
                                  summary=self.summary)
            report.generate_and_save()

        with open(self.output_file_data, 'w') as f:
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
        compartments = {key: [] for key in model_ensemble[0].results.keys() if key not in ('t_list', 'county_metadata')}
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
            surge_start_idx = np.argwhere(m.results[compartment] > getattr(m, compartment_to_capacity_attr_map[compartment]))
            surge_start.append(m.t_list[surge_start_idx[0][0]] if len(surge_start_idx) > 0 else float('NaN'))

            # Reverse the t-list and capacity and do the same.
            surge_end_idx = np.argwhere(m.results[compartment][::-1] > getattr(m, compartment_to_capacity_attr_map[compartment]))
            surge_end.append(m.t_list[::-1][surge_end_idx[0][0]] if len(surge_end_idx) > 0 else float('NaN'))

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
            peak_data['peak_value_ci%i' % percentile] = np.percentile(values_at_peak_index, percentile).tolist()
            peak_data['peak_time_ci%i' % percentile] = np.percentile(peak_times, percentile).tolist()

        peak_data['peak_value_mean'] = np.mean(values_at_peak_index).tolist()
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
        outputs['t_list'] = model_ensemble[0].t_list.tolist()

        # ------------------------------------------
        # Calculate Confidence Intervals and Peaks
        # ------------------------------------------
        for compartment, value_stack in self._generate_compartment_arrays(model_ensemble).items():
            compartment_output = dict()

            # Compute percentiles over the ensemble
            for percentile in self.output_percentiles:
                outputs[compartment]['ci_%i' % percentile] = np.percentile(value_stack, percentile, axis=0).tolist()

            if compartment in compartment_to_capacity_attr_map:
                compartment_output['surge_start'], compartment_output['surge_start'] = self._get_surge_window(model_ensemble, compartment)
                compartment_output['capacity'] = [getattr(m, compartment_to_capacity_attr_map[compartment]) for m in model_ensemble]

            compartment_output.update(self._detect_peak_time_and_value(value_stack, outputs['t_list']))

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


def run_state(state, ensemble_kwargs):
    """
    Run the EnsembleRunner for each county in a state.

    Parameters
    ----------
    state: str
        State to run against.
    ensemble_kwargs: dict
        Kwargs passed to the EnsembleRunner object.
    """
    # Run the state level
    runner = EnsembleRunner(fips=us.states.lookup(state).fips, **ensemble_kwargs)
    runner.run_ensemble()

    # Run county level
    df = load_data.load_county_metadata()
    all_fips = df[df['state'].str.lower() == state.lower()].fips
    p = Pool()
    f = partial(_run_county, ensemble_kwargs=ensemble_kwargs)
    p.map(f, all_fips)
    p.close()
