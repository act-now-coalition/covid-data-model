import scipy
import numpy as np
import pandas as pd
from enum import Enum
from pyseir.testing.load_data import load_population_size, load_rt, load_projection
from sklearn.model_selection import ParameterGrid
from datetime import datetime, date, timedelta


PCR_COST = 250                    # Cost of PCR test per individual
PCR_SENSITIVITY = 0.7             # Sensitivity of PCR test
ANTIBODY_COST = 120               # Cost of antibody test per individual
ANTIBODY_SENSITIVITY = 0.5        # Sensitivity of antibody test
ANTIBODY_FALSE_POSITIVITY = 0.1   # False positivity of antibody test
MAX_PCR_AVAILABILITY = np.inf       # Maximum availability of PCR test per month
MAX_ANTIBODY_AVAILABILITY = np.inf # Maximum availability of antibody test
HOSPITALIZATION_COST = 3000       # Cost of hospitalization due to covid
MAX_PCR_COST_PER_MONTH = np.inf   # Maximum affordable PCR cost
SITE_CLOSURE_COST = 1000000        # Cost of site closure
FRAC_CONTACT_ACTIVE = 0.5         # Fraction of contact (ppl) that will be
                                  # active at the site (e.g. back to work)
DATE = datetime.today()
DATE_FORMAT = "%Y-%m-%d"



class Allocation(Enum):
    RANDOM = 'random'
    STRATIFIED = 'stratified'
    ADAPTIVE = 'adaptive'


class TestStrategySimulator:
    """
    Simulate testing strategy, and calculates following metrics:
    -

    Parameters
    ----------


    Attributes
    ----------
    """
    def __init__(self,
                 fips,
                 date=DATE,
                 pcr_cost=PCR_COST,
                 antibody_cost=ANTIBODY_COST,
                 pcr_sensitivity=PCR_SENSITIVITY,
                 antibody_sensitivity=ANTIBODY_SENSITIVITY,
                 antibody_false_positivity=ANTIBODY_FALSE_POSITIVITY,
                 hospitalization_cost=HOSPITALIZATION_COST,
                 allocation='random',
                 pcr_coverage=np.linspace(0, 1, 11),
                 antibody_coverage=np.linspace(0, 1, 11),
                 pcr_frequency=[1, 2, 4],
                 relative_contact_rate=1.5,
                 frac_contact_active=FRAC_CONTACT_ACTIVE,
                 pcr_max_availability_per_month=MAX_PCR_AVAILABILITY,
                 antibody_max_availability=MAX_ANTIBODY_AVAILABILITY,
                 max_pcr_cost_per_month=MAX_PCR_COST_PER_MONTH,
                 site_closure_cost=SITE_CLOSURE_COST
                 ):
        self.fips = fips

        self.date = datetime.strftime(date, DATE_FORMAT)

        self.N = load_population_size(fips)
        self.Rt = load_rt(fips)
        self.Rt = self._index_time_to_str(self.Rt)
        self.Rt *= relative_contact_rate
        self.outbreak_threshold = self._calculate_outbreak_threshold()
        self.projection = load_projection(fips)
        self.projection = self._time_window(self.projection)
        self.projection = self._index_time_to_str(self.projection)

        self.pcr_sensitivity = pcr_sensitivity
        self.pcr_cost = pcr_cost
        self.pcr_coverage = pcr_coverage
        self.pcr_frequency = pcr_frequency
        self.pcr_max_availability_per_month = pcr_max_availability_per_month

        self.antibody_cost = antibody_cost
        self.antibody_sensitivity = antibody_sensitivity
        self.antibody_coverage = antibody_coverage
        self.antibody_max_availability = antibody_max_availability
        self.antibody_false_positivity = antibody_false_positivity

        self.hospitalization_cost = hospitalization_cost
        self.site_closure_cost = site_closure_cost

        self.allocation = Allocation(allocation)
        self.frac_contact_active = frac_contact_active

        self.max_pcr_cost_per_month = max_pcr_cost_per_month

        self.results = None


    def _calculate_outbreak_threshold(self, R0=None, k=0.16):
        """
        Calculate Minimum number of prevalent cases to trigger an outbreak
        at the site.
        Ref:
        equation (1) in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3680036/

        Parameters
        ----------
        R0: float
            basic reproduction number.
        k: float
            Dispersion factor that controls the distribution of secondary
            transmissions, lower value indicates a higher level of
            heterogeneity in transmission rates and greater variation in the
            distribution of secondary transmissions.
            Empirical estimates of SARS is 0.16. Ref:
            https://www.nature.com/articles/nature04153

        Returns
        -------
        outbreak_threshold: float
            Minimum number of prevalent cases to trigger an outbreak at the
            site.
        """
        R0 = R0 or self.Rt.loc[self.date]

        # ref: equ 1 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3680036/
        outbreak_threshold = (1 / np.log(self.Rt.loc[self.date])) \
        * (0.334 + 0.689/k + (0.408 - 0.507/k)/R0 + (- 0.356 + 0.467/k)/(R0 **
           2))

        return outbreak_threshold


    def _index_time_to_str(self, df, format=None):
        """
        Datetime to string based on give format.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe whose DatetimeIndex to be changed to string
        format: str
            Format of the time string, e.g. "%Y-%m-%d"

        Returns
        -------
        df: pd.DataFrame
            Dataframe index of which is string type and has given time format.
        """
        format = format or DATE_FORMAT
        df.index = df.index.strftime(format)
        return df

    def _time_window(self, df, date=None, window=30):
        """
        Constrain the time window of model projections to be within next
        n days of the start date.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with DatetimeIndex
        date: str or datetime
            Beginning date of the time window
        window: float or int
            Length of the time window in days.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with rows constrained to those with index that
            falls within next n days of the start date.
        """
        date = date or self.date
        if isinstance(self.date, str):
            date = datetime.strptime(self.date, DATE_FORMAT)

        within_time_window = [t for t in df.index if
                              t >= date and t <= date + timedelta(
                              days=window)]
        df = df.loc[within_time_window]
        return df


    def _calculate_outbreak_prob(self, prevalence):
        """
        Calculate probability of outbreak within next month, which depends on
        the maximum possible number of cases in the company in the coming
        month and the threshold number of cases per month that triggers
        outbreak (site closure).


        Parameters
        ----------
        reduced_case_num: float or int
            Reduced number of prevalent cases.

        Returns
        -------
        outbreak_prob: float
            Probability of outbreak within one month time window.
        """

        if prevalence > 0:
            outbreak_prob = 1 - scipy.stats.binom(n=self.N, p=prevalence).cdf(
                self.outbreak_threshold)

        else:
            outbreak_prob = 0

        return outbreak_prob


    def _calculate_infection_rates(self, prevalence, freq):
        """
        Calculates average prevalence within the time window covered by
        PCR tests.

        Parameters
        ----------
        freq: int
            Frequency of PCR tests per month


        Returns
        -------
        prevalence:
            Average prevalence throughout the time window of PCR tests.
        """
        freq = int(freq)
        if freq > 1:
            return prevalence[:int((freq - 1)*30/freq)]
        else:
            return prevalence[0]

    def run(self):
        """

        """
        param_grid = {'pcr_coverage': self.pcr_coverage,
                      'antibody_coverage': self.antibody_coverage,
                      'pcr_frequency': self.pcr_frequency}

        results = pd.DataFrame(list(ParameterGrid(param_grid)))

        # calculates the infection rates corresponding to all rounds of
        # pcr tests
        pcr_frequencies = results[['pcr_frequency']].drop_duplicates()
        prevalence_traj = self.projection['I'] / self.projection['N']

        # calculates average
        pcr_frequencies['pcr_covered_infection_rates'] = \
            pcr_frequencies.apply(
                lambda r: self._calculate_infection_rates(
                    prevalence_traj,
                    r.pcr_frequency).mean(),
                axis=1)

        results = pd.merge(results, pcr_frequencies, on='pcr_frequency')

        results['delta_p_infected'] = \
            results['pcr_coverage'] \
          * self.pcr_sensitivity \
          * results['pcr_covered_infection_rates'] \
          * results['pcr_frequency']

        results['delta_immunity'] = (self.projection['R'] / self.projection['N']).loc[self.date]\
                                  * (1 - self.frac_contact_active) \
                                  * results['antibody_coverage'] * self.antibody_sensitivity
        # assuming cases are detected and quarantined in the middle of
        # infectious period and spend half of their contacts at the site.
        results['prevented_secondary_transmission'] = \
            self.N * self.Rt.loc[self.date] * results['delta_p_infected'] / 2
        results['delta_covid_index'] = results['prevented_secondary_transmission'] \
                                     * self.projection[
                                           'IHR__per_capita'].loc[self.date] \
                                       / self.N
        results['test_cost_pcr'] = self.N * results['pcr_coverage'] * self.pcr_cost * results['pcr_frequency']

        # maximum prevalence during next month
        prevalence_no_testing = (self.projection['I'] / self.projection['N']).max()

        # TODO @Xinyu: refine this since this may overestimate the impact of
        # quarantine as majority of the cases will not last for entire month
        results['prevalence_given_testing'] = \
            prevalence_no_testing - (
                    results['delta_p_infected'] +
                    results['prevented_secondary_transmission']/self.N
            )
        results['prevalence_given_testing'] = results['prevalence_given_testing'].clip(lower=0)

        outbreak_prob_no_testing = self._calculate_outbreak_prob(prevalence_no_testing)

        results['outbreak_prob'] = results.apply(
            lambda r: self._calculate_outbreak_prob(
                prevalence=r.prevalence_given_testing), axis=1)

        results['delta_outbreak_prob'] = outbreak_prob_no_testing - results['outbreak_prob']
        results['saved_cost_of_site_closure'] = results['delta_outbreak_prob']\
                                               * self.site_closure_cost
        results['delta_hospitalization_cost'] = results['delta_covid_index'] * self.N * self.hospitalization_cost

        results['avoided_cost'] = results['saved_cost_of_site_closure'] + results['delta_hospitalization_cost']

        results['hospitalization_cost_no_test'] = self.N * (self.projection['I'] * self.projection['IHR__per_capita'] /
                                                  self.projection['N']).loc[self.date] * self.hospitalization_cost
        results['worksite_outbreak_cost_no_test'] = outbreak_prob_no_testing * self.site_closure_cost
        results['cost_no_test'] = results['hospitalization_cost_no_test'] + results['worksite_outbreak_cost_no_test']
        results['net_cost'] = results['cost_no_test'] + results['test_cost_pcr'] - results['avoided_cost']

        under_pcr_max_capacity = self.N * results['pcr_coverage'] * results['pcr_frequency'] \
                                 <= self.pcr_max_availability_per_month
        under_ab_max_capacity = self.N * results['antibody_coverage'] <= self.antibody_max_availability
        under_max_pcr_cost = results['test_cost_pcr'] <= self.max_pcr_cost_per_month

        self.results = results[under_pcr_max_capacity & under_ab_max_capacity & under_max_pcr_cost]

        return self.results


    def optimize(self, by='net_cost'):
        if self.results is None:
            self.results = self.run()

        if isinstance(by, str):
            by = [by]

        ascending = {'net_cost': True,
                     'delta_p_infected': False,
                     'outbreak_prob': True,
                     'test_cost_pcr': True}

        results = self.results.sort_values(by,
                                           ascending=[ascending[k] for k in
                                                      by]) \
                              .iloc[0].to_dict()

        return results

    def reformat_optimization_results(self, results):
        testing_parameters = ['antibody_coverage', 'pcr_coverage', 'pcr_frequency']
        metric_names = ['delta_p_infected',
                        'prevented_secondary_transmission',
                        'delta_covid_index',
                        'test_cost_pcr',
                        'prevalence_given_testing',
                        'outbreak_prob',
                        'delta_outbreak_prob',
                        'saved_cost_of_site_closure',
                        'avoided_cost']
        params = {k: v for k, v in results.items() if k in testing_parameters}
        metrics = {k: v for k, v in results.items() if k in metric_names}

        params = pd.DataFrame(params, index=['optimal'])
        metrics = pd.DataFrame(metrics, index=['metrics'])

        params = params.rename(columns={col: col + ' per month' for col in params.columns if 'pcr' in col}).T
        metrics = metrics.rename(columns={col: col + ' per month' for col in metrics.columns}).T

        return params, metrics




