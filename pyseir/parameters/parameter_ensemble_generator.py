import yaml
import os
import numpy as np
import pandas as pd
from pyseir import load_data
from pyseir.parameters.distribution import distribution
from libs.datasets import FIPSPopulation
from libs.datasets import DHBeds
import us


beds_data = None
population_data = None

THIS_FILE_PATH = os.path.dirname(os.path.abspath('__file__'))
DEFAULT_CONFIG = yaml.safe_load(open(os.path.join(THIS_FILE_PATH, 'defaults_config.yaml')).read())

class ParameterEnsembleGenerator:
    """
    Generate ensembles of parameters for SEIR modeling.

    Parameters
    ----------
    fips: str
        County or state fips code.
    N_samples: int
        Integer number of samples to generate.
    t_list: array-like
        Array of times to integrate against.
    I_initial: int
        Initial infected case count to consider.
    suppression_policy: callable(t): pyseir.model.suppression_policy
        Suppression policy to apply.
    parameter_defaults_config : dict
        Contains configurations that define the default distribution, value, functions to sample parameters.
        Example:
            R0:     # parameter to sample from a distribution
                func: ~
                clip: ~
                dist: uniform
                dist_params:
                    low: 3
                    high: 4.5
            mortality_rate_no_ventilator:   # parameter with fixed value
                value: 1
            A_initial:       # parameter as function of other parameters
                func:
                    value: 'lambda x, gamma, I_initial: gamma * I_initial / (1 - gamma)'
                    params:
                        - 'gamma'
                        - 'I_initial'
    """

    def __init__(self, fips, N_samples, t_list,
                 I_initial=1, suppression_policy=None,
                 parameter_defaults_config=DEFAULT_CONFIG):

        # Caching globally to avoid relatively significant performance overhead
        # of loading for each county.
        global beds_data, population_data
        if not beds_data or not population_data:
            beds_data = DHBeds.local().beds()
            population_data = FIPSPopulation.local().population()

        self.fips = fips
        self.geographic_unit = 'county' if len(self.fips) == 5 else 'state'
        self.N_samples = N_samples
        self.I_initial = I_initial
        self.suppression_policy = suppression_policy
        self.t_list = t_list

        if self.geographic_unit == 'county':
            self.county_metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
            self.state_abbr = us.states.lookup(self.county_metadata['state']).abbr
            self.population = population_data.get_county_level('USA', state=self.state_abbr, fips=self.fips)
            # TODO: Some counties do not have hospitals. Likely need to go to HRR level..
            self.beds = beds_data.get_county_level(self.state_abbr, fips=self.fips) or 0
        else:
            self.state_abbr = us.states.lookup(fips).abbr
            self.population = population_data.get_state_level('USA', state=self.state_abbr)
            self.beds = beds_data.get_state_level(self.state_abbr) or 0

        self.parameter_defaults_config = parameter_defaults_config

    def sample_parameter(self, param_config, args=None):
        """
        Sample parameter from distribution specified by the config.

        Parameters
        ----------
        param_config: dict
            Contains configurations that define the default distribution, value, functions to a sample for the
            parameter.
            Examples:
                R0:     # parameter to sample from a distribution
                    func: ~
                    clip: ~
                    dist: uniform
                    dist_params:
                        low: 3
                        high: 4.5

                mortality_rate_no_ventilator:   # parameter with fixed value
                    value: 1

                A_initial:       # parameter as function of other parameters
                    func:
                        value: 'lambda x, gamma, I_initial: gamma * I_initial / (1 - gamma)'
                        params:
                            - 'gamma'
                            - 'I_initial'
        args: dict
            Extra arguments to pass to 'func' of the config to generate parameter value.
            Keys should contain names specified in param_config['func']['params']

        Returns
        -------
        value : float
            Sampled parameter value.
        """
        args = args or {}
        value = None
        if 'value' in param_config:
            value = param_config['value']

        if 'dist' in param_config:
            dist = distribution(param_config['dist'])
            if dist == distribution.NORMAL:
                value = np.random.normal(**param_config['dist_params'])
            elif dist == distribution.GAMMA:
                value = np.random.gamma(**param_config['dist_params'])
            elif dist == distribution.UNIFORM:
                value = np.random.uniform(**param_config['dist_params'])
            elif dist == distribution.EXPONENTIAL:
                value = np.random.uniform(**param_config['dist_params'])

        if 'func' in param_config:
            if param_config['func']:
                func = eval(param_config['func']['value'])
                if 'params' in param_config['func']:
                    d = {k: v for k, v in args.items() if k in param_config['func']['params']}
                    value = func(value, **d)
                else:
                    value = func(value)

        if 'clip' in param_config:
            if param_config['clip']:
                value = np.clip(value, a_min=param_config['clip'][0], a_max=param_config['clip'][1])

        return value

    def sample_seir_parameters(self, override_params=None):
        """
        Generate N_samples of parameter values from the priors listed below.

        Parameters
        ----------
        override_params: dict()
            Individual parameters can be overridden here.

        Returns
        -------
        : list(dict)
            List of parameter sets to feed to the simulations.
        """
        override_params = override_params or dict()
        parameter_sets = []
        for _ in range(self.N_samples):

            # https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html
            # TODO: 10% is being used by CA group.  CDC suggests 20% case hospitalization rate
            # Note that this is 10% of symptomatic cases, making overall hospitalization around 5%.
            # https: // www.statista.com / statistics / 1105402 / covid - hospitalization - rates - us - by - age - group /
            parameter_set = dict(I_initial = self.I_initial,
                                    t_list=self.t_list,
                                    N=self.population,
                                    suppression_policy=self.suppression_policy)
            for param in self.parameter_defaults_config:
                parameter_set[param] = self.sample_parameter(self.parameter_defaults_config[param],
                                                             {**parameter_set, **self.__dict__})
            parameter_sets.append(parameter_set)

        for parameter_set in parameter_sets:
            parameter_set.update(override_params)

        return parameter_sets

    def get_average_seir_parameters(self):
        """
        Sample from the ensemble to obtain the average parameter values.

        Returns
        -------
        average_parameters: dict
            Average of the parameter ensemble, determined by sampling.
        """
        df = pd.DataFrame(self.sample_seir_parameters()).drop('t_list', axis=1)
        average_parameters = df.mean().to_dict()
        average_parameters['t_list'] = self.t_list
        return average_parameters
