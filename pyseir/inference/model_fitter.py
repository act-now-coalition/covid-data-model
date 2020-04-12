import logging
import iminuit
import numpy as np
import os
import us
import pandas as pd
import pprint
from copy import deepcopy
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from multiprocessing import Pool
from pyseir.models import suppression_policies
from pyseir import load_data, OUTPUT_DIR
from pyseir.models.seir_model import SEIRModel
from libs.datasets.dataset_utils import AggregationLevel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator


class ModelFitter:
    """
    Fit a SEIR model and suppression policy for a geographic unit (county or
    state) against case, hospitalization, and mortality data.  The error model
    to blend these is dominated by systematic uncertainties rather than
    statistical ones. We allow for relative rates of confirmed cases, hosp and
    deaths to float, allowing the fitter to focus on the shape parameters rather
    than absolutes.

    TODO: Add hospitalization data when available.

    Parameters
    ----------
    fips: str
        State or county fips code.
    ref_date: datetime
        Date to reference against. This should be before the first case.
    min_deaths: int
        Minimum number of fatalities to use death data.
    n_years: int
        Number of years to run the simulation for.
    """

    def __init__(self,
                 fips,
                 ref_date=datetime(year=2020, month=1, day=1),
                 min_deaths=5,
                 n_years=1):

        self.fips = fips
        self.ref_date = ref_date
        self.min_deaths = min_deaths
        self.t_list = np.linspace(0, int(365 * n_years), int(365 * n_years) + 1)

        if len(fips) == 2: # State FIPS are 2 digits
            self.agg_level = AggregationLevel.STATE
            self.state = us.states.lookup(self.fips).name
            self.geo_metadata = load_data.load_county_metadata_by_state(self.state).loc[self.state].to_dict()
            self.times, self.observed_new_cases, self.observed_new_deaths = \
                load_data.load_new_case_data_by_state(self.state, self.ref_date)
            self.display_name = self.state
        else:
            self.agg_level = AggregationLevel.COUNTY
            self.geo_metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
            self.state = self.geo_metadata['state']
            self.county = self.geo_metadata['county']
            self.display_name = self.county + ', ' + self.state
            # TODO Swap for new data source.
            self.times, self.observed_new_cases, self.observed_new_deaths = \
                load_data.load_new_case_data_by_fips(self.fips, t0=self.ref_date)

        self.cases_stdev, self.deaths_stdev = self.calculate_observation_errors()

        self.fit_params = dict(
            R0=3.0, limit_R0=[1, 8], error_R0=1.,
            I_initial=50.0, limit_I_initial=[1, 1e4], error_I_initial=10,
            t0=60, limit_t0=[-90, 90], error_t0=1,
            eps=.5, limit_eps=[0, 2], error_eps=.2,
            t_break=45, limit_t_break=[0, 100], error_t_break=5,
            test_fraction=.1, limit_test_fraction=[0.01, 1], error_test_fraction=.05,
            errordef=5
        )
        self.model_fit_keys = ['R0', 'eps', 't_break', 'I_initial']

        # self.suppression_policy_params = dict(t_list=self.t_list, reference_start_date=self.ref_date)
        self.SEIR_kwargs = self.get_average_SEIR_parameters()
        self.minuit = None
        self.fit_results = None
        self.mle_model = None

    def get_average_SEIR_parameters(self):
        """
        Generate the additional fitter candidates from the ensemble generator. This
        has the suppression policy and R0 keys removed.

        Returns
        -------
        SEIR_kwargs: dict
            The average ensemble params.
        """
        SEIR_kwargs = ParameterEnsembleGenerator(
            fips=self.fips,
            N_samples=5000,
            t_list=self.t_list,
            suppression_policy=None).get_average_seir_parameters()

        SEIR_kwargs = {k: v for k, v in SEIR_kwargs.items() if k not in self.fit_params}
        del SEIR_kwargs['suppression_policy']
        return SEIR_kwargs

    def calculate_observation_errors(self):
        """
        Generate the errors on the observations. Statistical errors are more
        or less irrelevant to to the systematic uncertainties associated with
        under-testing, under-counting of deaths and hospitalizations etc..

        Assume the error on the case count could be off by a massive factor
        50. Basically we don't want to use it if there appreciable mortality
        data available. Longer term there is a better procedure. Make the
        stdev N times larger x the number of cases.
        
        Returns
        -------
        cases_stdev: array-like
            Float uncertainties (stdev) for case data.
        deaths_stdev: array-like
            Float uncertainties (stdev) for death data.
        """
        # This basically ignores cases...
        cases_stdev = 1e5 * self.observed_new_cases.copy()
        # stdev = 200% of the values...
        deaths_stdev = 2 * self.observed_new_deaths.copy()

        # Zero inflated poisson Avoid floating point errors..
        cases_stdev[cases_stdev == 0] = 1e10
        deaths_stdev[deaths_stdev == 0] = 1e10

        return cases_stdev, deaths_stdev

    def run_model(self, R0, eps, t_break, I_initial):
        """
        Generate the model.

        Returns
        -------
        model: SEIRModel
        """
        # if by == 'fips':
        #     suppression_policy = \
        #         suppression_policies.generate_empirical_distancing_policy(
        #             fips=fips, future_suppression=eps, **suppression_policy_params)
        # elif by == 'state':
        #     # TODO: This takes > 200ms which is 10x the model run time. Can be optimized...
        #     suppression_policy = \
        #         suppression_policies.generate_empirical_distancing_policy_by_state(
        #             state=state, future_suppression=eps, **suppression_policy_params)

        suppression_policy = suppression_policies.generate_two_step_policy(self.t_list, eps, t_break)
        self.SEIR_kwargs['E_initial'] = 1.2 * I_initial

        model = SEIRModel(
            R0=R0,
            suppression_policy=suppression_policy,
            I_initial=I_initial,
            **self.SEIR_kwargs)
        model.run()
        return model

    def _fit_seir(self, R0, t0, eps, t_break, test_fraction, I_initial):
        """
        Fit SEIR model by MLE.

        Parameters
        ----------
        R0: float
            Basic reproduction number
        t0: float
            Epidemic starting time.
        eps:
            Fraction of reduction in contact rates as result of  to suppression
            policy projected into future.
        suppression_policy_params: dict
            Parameters to pass to suppression policy model.

        Returns
        -------
          : float
            Chi square of fitting model to observed cases and deaths.
        """
        l = locals()
        model_kwargs = {k: l[k] for k in self.model_fit_keys}
        model = self.run_model(**model_kwargs)

        # Extract the predicted rates from the model.
        predicted_cases = (test_fraction * model.gamma
                           * np.interp(self.times, self.t_list + t0, model.results['total_new_infections']))

        predicted_deaths = np.interp(self.times, self.t_list + t0, model.results['total_deaths_per_day'])

        # Compute Chi2
        chi2_cases = np.sum((self.observed_new_cases - predicted_cases) ** 2 / self.cases_stdev ** 2)

        # Only use deaths if there are enough observations..
        if self.observed_new_deaths.sum() > self.min_deaths:
            chi2_deaths = np.sum((self.observed_new_deaths - predicted_deaths) ** 2 / self.deaths_stdev ** 2)
        else:
            chi2_deaths = 0

        logging.info(f'Chi2 Cases {chi2_cases:1.2f}, Death {chi2_deaths:1.2f}')
        return chi2_deaths + chi2_cases

    def fit(self):
        """
        Fit a model to the data.
        """
        self.minuit = iminuit.Minuit(self._fit_seir, **self.fit_params)

        # run MIGRAD algorithm for optimization.
        # for details refer: https://root.cern/root/html528/TMinuit.html
        # TODO @ Xinyu: add lines to check if minuit optimization result is valid.
        self.minuit.migrad()

        self.fit_results = dict(fips=self.fips, **dict(self.minuit.values))

        if np.isnan(self.fit_results['t0']):
            logging.error(f'Could not compute MLE values for {self.display_name}')
            self.fit_results['t0_date'] = self.ref_date + timedelta(days=self.t0_guess)
        else:
            self.fit_results['t0_date'] = self.ref_date + timedelta(days=self.fit_results['t0'])
        self.fit_results['Reff'] = self.fit_results['R0'] * self.fit_results['eps']
        self.fit_results['name'] = self.display_name
        self.fit_results['total_population'] = self.geo_metadata['total_population']
        self.fit_results['population_density'] = self.geo_metadata['population_density']
        self.fit_results['valid_fit'] = self.minuit.migrad_ok()
        logging.info(f'Fit Results for {self.display_name} \n {pprint.pformat(self.fit_results)}')

        self.mle_model = self.run_model(**{k: self.fit_results[k] for k in self.model_fit_keys})

    def plot_fitting_results(self):
        """
        Plotting model fitting results.
        """
        data_dates = [self.ref_date + timedelta(days=t) for t in self.times]
        model_dates = [self.ref_date + timedelta(days=t + self.fit_results['t0']) for t in self.t_list]

        # Don't display the zero-inflated error bars
        cases_err = np.array(self.deaths_stdev)
        cases_err[cases_err > 1e5] = 0
        death_err = deepcopy(self.deaths_stdev)
        death_err[death_err > 1e5] = 0

        plt.figure(figsize=(10, 8))
        plt.errorbar(data_dates, self.observed_new_cases, yerr=cases_err,
                     marker='o', linestyle='', label='Observed Cases Per Day', color='steelblue', capsize=3)


        plt.errorbar(data_dates, self.observed_new_deaths, yerr=death_err,
                     marker='d', linestyle='', label='Observed Deaths Per Day', color='firebrick', capsize=3)
        plt.plot(model_dates, self.mle_model.results['total_new_infections'],
                 label='Estimated Total New Infections Per Day', linestyle='--', lw=2, color='steelblue')
        plt.plot(model_dates, self.fit_results['test_fraction'] * self.mle_model.results['total_new_infections'],
                 label='Estimated Tested New Infections Per Day', color='steelblue')
        plt.plot(model_dates, self.mle_model.results['total_deaths_per_day'],
                 label='Model Deaths Per Day', color='firebrick')

        plt.yscale('log')
        plt.ylim(.8e0)
        plt.xlim(data_dates[0], data_dates[-1] + timedelta(days=90))

        plt.xticks(rotation=30)
        plt.legend(loc=1)
        plt.grid(which='both', alpha=.3)
        plt.title(self.display_name)

        for i, (k, v) in enumerate(self.fit_results.items()):
            if k not in ('fips', 't0_date', 'display_name'):
                plt.text(.025, .97 - 0.04 * i, f'{k}={v:1.3f}',
                         transform=plt.gca().transAxes, fontsize=12)
            else:
                plt.text(.025, .97 - 0.04 * i, f'{k}={v}',
                         transform=plt.gca().transAxes, fontsize=12)

        if self.agg_level is AggregationLevel.COUNTY:
            output_file = os.path.join(OUTPUT_DIR, 'pyseir', self.fit_results['state'].title(), 'reports',
                f'{self.state}__{self.county}__{self.fips}__mle_fit_results.pdf')
        else:
            output_file = os.path.join(
                OUTPUT_DIR, 'pyseir', 'state_summaries', f'{self.state}__{self.fips}__mle_fit_results.pdf')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)

    @classmethod
    def run_for_fips(cls, fips):
        """
        Run the model fitter for a state or county fips code.

        Parameters
        ----------
        fips: str
            2-digit state or 5-digit county fips code.

        Returns
        -------
        : ModelFitter

        """
        model_fitter = cls(fips)
        model_fitter.fit()
        model_fitter.plot_fitting_results()
        return model_fitter


def run_state(state, states_only=False, case_death_timeseries=None):
    """
    Run the fitter for each county in a state.

    Parameters
    ----------
    state: str
        State to run against.
    states_only: bool
        If True only run the state level.
    case_death_timeseries: Timeseries
        Case, death, timeseries.
    """
    state_obj = us.states.lookup(state)
    logging.info(f'Running MLE fitter for state {state_obj.name}')
    state_output_file = os.path.join(OUTPUT_DIR, 'pyseir', 'data', 'state_summary',
                               f'summary_{state}_state_only__mle_fit_results.json')
    os.makedirs(os.path.dirname(state_output_file), exist_ok=True)
    fit_results = ModelFitter.run_for_fips(state_obj.fips)
    pd.DataFrame(fit_results, index=[state]).to_json(state_output_file)

    # Run the counties.
    if not states_only:
        county_output_file = os.path.join(OUTPUT_DIR, 'pyseir', 'data', 'state_summary',
                                   f'summary_{state}__mle_fit_results.json')

        df = load_data.load_county_metadata()
        all_fips = df[df['state'].str.lower() == state_obj.name.lower()].fips

        p = Pool()
        fitters = p.map(ModelFitter.run_for_fips, all_fips)
        p.close()

        # Output
        pd.DataFrame([fit.fit_results for fit in fitters]).to_json(county_output_file)

