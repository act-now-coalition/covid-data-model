import logging
import iminuit
import numpy as np
import os
import us
from pprint import pformat
import pandas as pd
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
    cases_to_deaths_err_factor: float
        To control chi2 mix between cases and deaths, we multiply the systematic
        errors of cases by this number. Thus higher numbers reduce the influence
        of case data on the fit.
    hospital_to_deaths_err_factor: float
        To control chi2 mix between hospitalization and deaths, we multiply the
        systematic errors of cases by this number. Thus higher numbers reduce
        the influence of hospitalization data on the fit.
    """
    def __init__(self,
                 fips,
                 ref_date=datetime(year=2020, month=1, day=1),
                 min_deaths=5,
                 n_years=1,
                 cases_to_deaths_err_factor=3,
                 hospital_to_deaths_err_factor=1):

        self.fips = fips
        self.ref_date = ref_date
        self.min_deaths = min_deaths
        self.t_list = np.linspace(0, int(365 * n_years), int(365 * n_years) + 1)
        self.cases_to_deaths_err_factor = cases_to_deaths_err_factor
        self.hospital_to_deaths_err_factor = hospital_to_deaths_err_factor
        self.t0_guess = 60

        if len(fips) == 2: # State FIPS are 2 digits
            self.agg_level = AggregationLevel.STATE
            self.state_obj = us.states.lookup(self.fips)
            self.state = self.state_obj.name
            self.geo_metadata = load_data.load_county_metadata_by_state(self.state.title()).loc[self.state.title()].to_dict()
            self.times, self.observed_new_cases, self.observed_new_deaths = \
                load_data.load_new_case_data_by_state(self.state, self.ref_date)
            self.hospital_times, self.hospitalizations, self.hospitalization_data_type = \
                load_data.load_hospitalization_data_by_state(self.state_obj.abbr, t0=self.ref_date)
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
            self.hospital_times, self.hospitalizations, self.hospitalization_data_type = \
                load_data.load_hospitalization_data(self.fips, t0=self.ref_date)

        self.cases_stdev, self.hosp_stdev, self.deaths_stdev = self.calculate_observation_errors()

        self.fit_params = dict(
            R0=3.4, limit_R0=[1, 8], error_R0=1.,
            log10_I_initial=2, limit_log10_I_initial=[0, 5], error_log10_I_initial=.5,
            t0=60, limit_t0=[0, 70], error_t0=1,
            eps=.3, limit_eps=[0, 2], error_eps=.2,
            t_break=20, limit_t_break=[0, 100], error_t_break=2,
            test_fraction=.05, limit_test_fraction=[0.001, 1], error_test_fraction=.01,
            hosp_fraction=1, limit_hosp_fraction=[0.001, 1], error_hosp_fraction=.1,
            fix_hosp_fraction=self.hospitalizations is None, # Let's not fit this to start...
            errordef=.5
        )
        self.model_fit_keys = ['R0', 'eps', 't_break', 'log10_I_initial']

        self.SEIR_kwargs = self.get_average_seir_parameters()
        self.minuit = None
        self.fit_results = None
        self.mle_model = None

        self.chi2_deaths = None
        self.chi2_cases = None
        self.chi2_hosp = None
        self.dof_deaths = None
        self.dof_cases = None
        self.dof_hosp = None

    def get_average_seir_parameters(self):
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
        del SEIR_kwargs['I_initial']
        return SEIR_kwargs

    def calculate_observation_errors(self):
        """
        Generate the errors on the observations.

        Here we throw out a few assumptions to plant a flag...

        1. Systematic errors dominate and are likely of order 100% based on 100%
           undercounting of deaths and hospitalizations in many places.
        2. 100% is too small for 1 case count or mortality.. We should be much
           more confident in large numbers of observations
        3. TODO: Deal with this fact.. Actual observations are lower bounds.
                 Need asymmetric errors.

        As an error model, absolutes are less important to our problem compared
        to getting relative error scaling reasonably done. This is not true if
        drawing contours and confidence intervals which is why we choose large
        conservative errors to overestimate the uncertainty.

        As a relative scaling model we think about Poisson processes and scale
        the errors in the following way:

        1. Set the error of the largest observation to 100% of its value.
        2. Scale all other errors based on sqrt(value) * sqrt(max_value)
        
        Returns
        -------
        cases_stdev: array-like
            Float uncertainties (stdev) for case data.
        hosp_stdev: array-like
            Float uncertainties (stdev) for hosp data.
        deaths_stdev: array-like
            Float uncertainties (stdev) for death data.
        """
        # Stdev 200% of values.
        cases_stdev = self.cases_to_deaths_err_factor * self.observed_new_cases ** 0.5 * self.observed_new_cases.max() ** 0.5
        deaths_stdev = self.observed_new_deaths ** 0.5 * self.observed_new_deaths.max() ** 0.5

        # If cumulative hospitalizations, differentiate.
        if self.hospitalization_data_type == 'cumulative':
            hosp_data = self.hospitalizations[1:] - self.hospitalizations[:-1]
            hosp_stdev = self.hospital_to_deaths_err_factor * hosp_data ** 0.5 * hosp_data.max() ** 0.5
        elif self.hospitalization_data_type == 'current':
            hosp_data = self.hospitalizations
            hosp_stdev = self.hospital_to_deaths_err_factor * hosp_data ** 0.5 * hosp_data.max() ** 0.5
        else:
            hosp_stdev = None

        # Zero inflated poisson Avoid floating point errors..
        cases_stdev[cases_stdev == 0] = 1e10
        deaths_stdev[deaths_stdev == 0] = 1e10
        if hosp_stdev is not None:
            hosp_stdev[hosp_stdev == 0] = 1e10

        return cases_stdev, hosp_stdev, deaths_stdev

    def run_model(self, R0, eps, t_break, log10_I_initial):
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
        self.SEIR_kwargs['E_initial'] = 1.2 * 10**log10_I_initial

        model = SEIRModel(
            R0=R0,
            suppression_policy=suppression_policy,
            I_initial=10**log10_I_initial,
            **self.SEIR_kwargs)
        model.run()
        return model

    def _fit_seir(self, R0, t0, eps, t_break, test_fraction, hosp_fraction, log10_I_initial):
        """
        Fit SEIR model by MLE.

        Parameters
        ----------
        R0: float
            Basic reproduction number
        t0: float
            Epidemic starting time.
        eps: float
            Fraction of reduction in contact rates as result of  to suppression
            policy projected into future.
        t_break: float
            Timing for the switch in suppression policy.
        test_fraction: float
            Fraction of cases that get tested.
        hosp_fraction: float
            Fraction of actual hospitalizations vs the total.
        log10_I_initial:
            log10 initial infections.

        Returns
        -------
          : float
            Chi square of fitting model to observed cases and deaths.
        """
        l = locals()
        model_kwargs = {k: l[k] for k in self.model_fit_keys}
        model = self.run_model(**model_kwargs)

        #-----------------------------------
        # Chi2 Cases
        # -----------------------------------
        # Extract the predicted rates from the model.
        predicted_cases = (test_fraction * model.gamma
                           * np.interp(self.times, self.t_list + t0, model.results['total_new_infections']))
        chi2_cases = np.sum((self.observed_new_cases - predicted_cases) ** 2 / self.cases_stdev ** 2)

        # -----------------------------------
        # Chi2 Hospitalizations
        # -----------------------------------
        if self.hospitalization_data_type == 'current':
            predicted_hosp = hosp_fraction * np.interp(self.hospital_times, self.t_list + t0,
                                                       model.results['HGen'] + model.results['HICU'])
            chi2_hosp = np.sum((self.hospitalizations - predicted_hosp) ** 2 / self.hosp_stdev ** 2)
            self.dof_hosp = (self.observed_new_cases > 0).sum()
        elif self.hospitalization_data_type == 'cumulative':
            # Cumulative, so differentiate the data
            cumulative_hosp_predicted = model.results['HGen_cumulative'] + model.results['HICU_cumulative']
            new_hosp_predicted = cumulative_hosp_predicted[1:] - cumulative_hosp_predicted[:-1]
            new_hosp_predicted = hosp_fraction * np.interp(self.hospital_times[1:], self.t_list[1:] + t0, new_hosp_predicted)
            new_hosp_observed = self.hospitalizations[1:] - self.hospitalizations[:-1]
            chi2_hosp = np.sum((new_hosp_observed - new_hosp_predicted) ** 2 / self.hosp_stdev ** 2)
            self.dof_hosp = (self.observed_new_cases > 0).sum()
        else:
            chi2_hosp = 0
            self.dof_hosp = 1e-10

        # -----------------------------------
        # Chi2 Deaths
        # -----------------------------------
        # Only use deaths if there are enough observations..
        predicted_deaths = np.interp(self.times, self.t_list + t0, model.results['total_deaths_per_day'])
        if self.observed_new_deaths.sum() > self.min_deaths:
            chi2_deaths = np.sum((self.observed_new_deaths - predicted_deaths) ** 2 / self.deaths_stdev ** 2)
        else:
            chi2_deaths = 0

        self.chi2_deaths = chi2_deaths
        self.chi2_cases = chi2_cases
        self.chi2_hosp = chi2_hosp
        self.dof_deaths = (self.observed_new_deaths > 0).sum()
        self.dof_cases = (self.observed_new_cases > 0).sum()

        return chi2_deaths + chi2_cases + chi2_hosp

    def fit(self):
        """
        Fit a model to the data.
        """
        self.minuit = iminuit.Minuit(self._fit_seir, **self.fit_params, print_level=1)

        # run MIGRAD algorithm for optimization.
        # for details refer: https://root.cern/root/html528/TMinuit.html
        # TODO @ Xinyu: add lines to check if minuit optimization result is valid.
        self.minuit.migrad(precision=1e-4)
        self.fit_results = dict(fips=self.fips, **dict(self.minuit.values))
        # This just updates chi2 values
        self._fit_seir(**dict(self.minuit.values))

        if np.isnan(self.fit_results['t0']):
            logging.error(f'Could not compute MLE values for {self.display_name}')
            self.fit_results['t0_date'] = self.ref_date + timedelta(days=self.t0_guess)
        else:
            self.fit_results['t0_date'] = self.ref_date + timedelta(days=self.fit_results['t0'])
        self.fit_results['Reff'] = self.fit_results['R0'] * self.fit_results['eps']

        self.fit_results['chi2/dof cases'] = self.chi2_cases / (self.dof_cases - 1)
        if self.hospitalizations is not None:
            self.fit_results['chi2/dof hosp'] = self.chi2_hosp / (self.dof_hosp - 1)
        self.fit_results['chi2/dof deaths'] = self.chi2_deaths / (self.dof_deaths - 1)

        try:
            param_state = self.minuit.get_param_states()
            logging.info(f'Fit Results for {self.display_name} \n {param_state}')
        except:
            param_state = dict(self.minuit.values)
            logging.info(f'Fit Results for {self.display_name} \n {param_state}')

        logging.info(f'Complete fit results for {self.display_name} \n {pformat(self.fit_results)}')
        self.mle_model = self.run_model(**{k: self.fit_results[k] for k in self.model_fit_keys})

    def plot_fitting_results(self):
        """
        Plotting model fitting results.
        """
        data_dates = [self.ref_date + timedelta(days=t) for t in self.times]
        if self.hospital_times is not None:
            hosp_dates = [self.ref_date + timedelta(days=float(t)) for t in self.hospital_times]
        model_dates = [self.ref_date + timedelta(days=t + self.fit_results['t0']) for t in self.t_list]

        # Don't display the zero-inflated error bars
        cases_err = np.array(self.cases_stdev)
        cases_err[cases_err > 1e5] = 0
        death_err = deepcopy(self.deaths_stdev)
        death_err[death_err > 1e5] = 0
        if self.hosp_stdev is not None:
            hosp_stdev = deepcopy(self.hosp_stdev)
            hosp_stdev[hosp_stdev > 1e5] = 0

        plt.figure(figsize=(18, 12))
        plt.errorbar(data_dates, self.observed_new_cases, yerr=cases_err,
                     marker='o', linestyle='', label='Observed Cases Per Day',
                     color='steelblue', capsize=3, alpha=.4, markersize=10)
        plt.errorbar(data_dates, self.observed_new_deaths, yerr=death_err,
                     marker='d', linestyle='', label='Observed Deaths Per Day',
                     color='firebrick', capsize=3, alpha=.4, markersize=10)

        plt.plot(model_dates, self.mle_model.results['total_new_infections'],
                 label='Estimated Total New Infections Per Day', linestyle='--', lw=4, color='steelblue')
        plt.plot(model_dates, self.fit_results['test_fraction'] * self.mle_model.results['total_new_infections'],
                 label='Estimated Tested New Infections Per Day', color='steelblue', lw=4)

        plt.plot(model_dates, self.mle_model.results['total_deaths_per_day'],
                 label='Model Deaths Per Day', color='firebrick', lw=4)

        if self.hospitalization_data_type == 'cumulative':
            new_hosp_observed = self.hospitalizations[1:] - self.hospitalizations[:-1]
            plt.errorbar(hosp_dates[1:], new_hosp_observed, yerr=hosp_stdev,
                         marker='s', linestyle='', label='Observed New Hospitalizations Per Day',
                         color='darkseagreen', capsize=3, alpha=1)
            predicted_hosp = (self.mle_model.results['HGen_cumulative'] + self.mle_model.results['HICU_cumulative'])
            predicted_hosp = predicted_hosp[1:] - predicted_hosp[:-1]
            plt.plot(model_dates[1:],
                     self.fit_results['hosp_fraction'] * predicted_hosp,
                     label='Estimated Total New Hospitalizations Per Day',
                     linestyle='-.', lw=4, color='darkseagreen', markersize=10)
        elif self.hospitalization_data_type == 'current':
            plt.errorbar(hosp_dates, self.hospitalizations, yerr=hosp_stdev,
                         marker='s', linestyle='', label='Observed Total Current Hospitalizations',
                         color='darkseagreen', capsize=3, alpha=.5, markersize=10)
            predicted_hosp = (self.mle_model.results['HGen'] + self.mle_model.results['HICU'])
            plt.plot(model_dates, self.fit_results['hosp_fraction'] * predicted_hosp,
                     label='Estimated Total Current Hospitalizations',
                     linestyle='-.', lw=4, color='darkseagreen')

        plt.plot(model_dates,
                 self.fit_results['hosp_fraction'] * self.mle_model.results['HICU'],
                 label='Estimated ICU Occupancy',
                 linestyle=':', lw=6, color='black')

        plt.yscale('log')
        plt.ylim(.8e0)
        plt.xlim(data_dates[0], data_dates[-1] + timedelta(days=150))

        plt.xticks(rotation=30, fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc=1, fontsize=14)
        plt.grid(which='both', alpha=.5)
        plt.title(self.display_name, fontsize=20)

        for i, (k, v) in enumerate(self.fit_results.items()):
            if np.isscalar(v) and not isinstance(v, str):
                plt.text(.7, .45 - 0.032 * i, f'{k}={v:1.3f}',
                         transform=plt.gca().transAxes, fontsize=15, alpha=.6)
            else:
                plt.text(.7, .45 - 0.032 * i, f'{k}={v}',
                         transform=plt.gca().transAxes, fontsize=15, alpha=.6)

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
    fit_results = ModelFitter.run_for_fips(state_obj.fips).fit_results

    pd.DataFrame(fit_results, index=[state_obj.fips]).to_json(state_output_file)

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

