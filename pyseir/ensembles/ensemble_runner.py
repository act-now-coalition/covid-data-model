import datetime
import logging
import os
import yaml
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

THIS_FILE_PATH = os.path.dirname(os.path.abspath('__file__'))
CONFIG = yaml.safe_load(open(os.path.join(THIS_FILE_PATH, 'config.yaml')).read())
DEFAULT_CONFIG = yaml.safe_load(open(os.path.join(THIS_FILE_PATH, '..', 'parameters', 'defaults_config.yaml')).read())

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
    config: dict
        Contains:
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
        override: dict
            configuration to override parameters
    parameter_defaults_config: dict
        Configuration to sample parameter from default distributions.
    """
    def __init__(self,
                 fips,
                 config=CONFIG,
                 parameter_defaults_config=DEFAULT_CONFIG):

        # Caching globally to avoid relatively significant performance overhead
        # of loading for each county.
        global jhu_timeseries
        if not jhu_timeseries:
            jhu_timeseries = JHUDataset.local().timeseries()

        self.fips = fips
        self.geographic_unit = 'county' if len(self.fips) == 5 else 'state'

        self.t_list = np.linspace(0, 365 * config['n_years'], 365 * config['n_years'])
        self.skip_plots = config['skip_plots']
        self.run_mode = RunMode(config['run_mode'])

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

        self.output_percentiles = config['output_percentiles']
        self.n_samples = config['n_samples']
        self.n_years = config['n_years']
        # TODO: Will be soon replaced with loaders for all the inferred params.
        # self.t0 = fit_results.load_t0(fips)
        self.date_generated = datetime.datetime.utcnow().isoformat()
        self.suppression_policy = config['suppression_policy']
        self.summary = copy.deepcopy(self.__dict__)
        self.summary.pop('t_list')
        self.generate_report = config['generate_report']
        self.override_params_config = config['override']
        self.parameter_defaults_config = parameter_defaults_config

        self.suppression_policies = None
        self.override_params = None
        self.init_run_mode()

        self.all_outputs = {}

    def override_parameter(self, override_param_config, args=None):
        """
        Get the value of override parameter from the config.

        Parameters
        ----------
        override_param_config: dict
            Contains configurations required to generate the value used to override the parameter.
            Examples:
                R0:
                  value: 3
                A_initial:
                func:
                  value: 'lambda x, cases: 1.0 * 3.43 * cases.max()
                             if (len(cases) > 0)
                             & (max(cases, default=-1) > 0)
                             else 1'
                  params:
                  - 'cases'
        args: dict
            Extra arguments to pass to 'func' of the config to generate override value.
            Keys should contain names specified in override_param_config['func']['params']

        Returns
        -------
        value : float
            Value to override parameter original value.
        """
        args = args or {}
        value = None
        if 'value' in override_param_config:
            value = override_param_config['value']

        if 'func' in override_param_config:
            if override_param_config['func']:
                func = eval(override_param_config['func']['value'])
                if 'params' in override_param_config['func']:
                    d = {k: v for k, v in args.items() if k in override_param_config['func']['params']}
                    value = func(value, **d)
                else:
                    value = func(value)

        return value

    def init_run_mode(self):
        """
        Based on the run mode, generate suppression policies and ensemble
        parameters.  This enables different model combinations and project
        phases.
        """
        self.suppression_policies = dict()
        self.override_params = dict()
        if self.run_mode is RunMode.CAN_BEFORE:
            self.n_samples = 1

            for scenario in ['no_intervention', 'flatten_the_curve', 'full_containment', 'social_distancing']:
                R0 = self.override_parameter(self.override_params_config['R0'],
                                             args={**self.override_params, **self.covid_data.__dict__})
                policy = generate_covidactnow_scenarios(t_list=self.t_list, R0=R0,
                                                        t0=datetime.datetime.today(), scenario=scenario)
                self.suppression_policies[f'suppression_policy__{scenario}'] = policy
                self.override_params = ParameterEnsembleGenerator(
                    self.fips, N_samples=500, t_list=self.t_list, suppression_policy=policy,
                    parameter_defaults_config=self.parameter_defaults_config
                ).get_average_seir_parameters()

            for param in self.override_params_config:
                self.override_params[param] = self.override_parameter(
                    self.override_params_config[param],
                    args={**self.override_params, **self.covid_data.__dict__})

            self.override_params = {k: v for k, v in self.override_params.items() if v is not None}
        elif self.run_mode is RunMode.DEFAULT:
            for suppression_policy in self.suppression_policy:
                self.suppression_policies[f'suppression_policy__{suppression_policy}']= generate_empirical_distancing_policy(
                    t_list=self.t_list, fips=self.fips, future_suppression=suppression_policy)
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
