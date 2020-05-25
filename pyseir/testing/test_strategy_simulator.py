import numpy as np
import pandas as pd
from enum import Enum
from pyseir.testing.load_data import load_population_size, load_rt, load_projection
from sklearn.model_selection import ParameterGrid
from datetime import datetime, date, timedelta
from tabulate import tabulate


PCR_COST = 250
PCR_SENSITIVITY = 0.7
ANTIBODY_COST = 120
ANTIBODY_SENSITIVITY = 0.5
ANTIBODY_FALSE_POSITIVITY = 0.1
MAX_PCR_AVAILABILITY = 1000
MAX_ANTIBODY_AVAILABILITY = 1000
MIN_NUM_CASE_OUTBREAK = 5 # minimum number of cases per month that triggers closure
HOSPITALIZATION_COST = 3000  # this maybe way off the real number
MAX_PCR_COST_PER_MONTH = 100000
DATE = datetime.today()
SITE_CLOSURE_COST = 200000
DATE_FORMAT = "%Y-%m-%d"
FRAC_CONTACT_ACTIVE = 0.5


class Allocation(Enum):
    RANDOM = 'random'
    STRATIFIED = 'stratified'
    ADAPTIVE = 'adaptive'


class TestStrategySimulator:
    """

    """
    def __init__(self,
                 fips,
                 date=DATE,
                 pcr_cost=PCR_COST,
                 antibody_cost=ANTIBODY_COST,
                 pcr_sensitivity=PCR_SENSITIVITY,
                 antibody_sensitivity=ANTIBODY_SENSITIVITY,
                 antibody_false_positivity=ANTIBODY_FALSE_POSITIVITY,
                 hospitalization_cost = HOSPITALIZATION_COST,
                 allocation='random',
                 pcr_coverage=np.linspace(0, 1, 11),
                 antibody_coverage=np.linspace(0, 1, 11),
                 pcr_frequency=[1, 2, 4],
                 relative_contact_rate=1,
                 frac_contact_active=FRAC_CONTACT_ACTIVE,
                 min_num_case_outbreak=MIN_NUM_CASE_OUTBREAK,
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
        self.projection = load_projection(fips)
        self.projection = self._index_time_to_str(self.projection)

        self.relative_contact_rate = relative_contact_rate

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
        self.min_num_case_outbreak = min_num_case_outbreak

        self.max_pcr_cost_per_month = max_pcr_cost_per_month

        self.results = None


    def _index_time_to_str(self, df):
        """

        """
        df.index = df.index.strftime(DATE_FORMAT)
        return df

    def _calculate_outbreak_prob(self, reduced_case_num=None):
        """
        Calculate probability of outbreak, which depends on the maximum possible number of cases in the company in
        the coming month.
        Probability = minimum(1, ratio of prevalent cases in company/minimum number of cases to trigger outbreak per
        day).
        """
        reduced_case_num = reduced_case_num or 0
        to_datetime = lambda x: datetime.strptime(x, DATE_FORMAT)
        within_next_month = [s for s in self.projection.index if to_datetime(s) >= to_datetime(self.date)
                             and to_datetime(s) <= to_datetime(self.date) + timedelta(days=30)]

        prevalent_cases = ((self.projection['I'] / self.projection['N']) * self.N).loc[
                           within_next_month].max()

        outbreak_prob = np.minimum(1, (prevalent_cases - reduced_case_num) / self.min_num_case_outbreak)

        return outbreak_prob

    def _calculate_infection_rates(self, freq):
        to_datetime = lambda x: datetime.strptime(x, DATE_FORMAT)
        within_next_month = [s for s in self.projection.index if to_datetime(s) >= to_datetime(self.date)
                             and to_datetime(s) <= to_datetime(self.date) + timedelta(days=30)]

        prevalence = (self.projection['I'] / self.projection['N']).loc[within_next_month].values
        indices = [int(i * (30 // freq) - 1) for i in range(1, int(freq + 1))][:int(freq)]
        prevalence = prevalence[indices]
        return prevalence

    def run(self):
        """

        """
        param_grid = {'pcr_coverage': self.pcr_coverage,
                      'antibody_coverage': self.antibody_coverage,
                      'pcr_frequency': self.pcr_frequency}

        results = pd.DataFrame(list(ParameterGrid(param_grid)))

        results['delta_p_infected'] = results['pcr_frequency'] * results['pcr_coverage'] * self.pcr_sensitivity * \
                                      (self.projection['I'] / self.projection['N']).loc[self.date]
        results['delta_immunity'] = (self.projection['R'] / self.projection['N']).loc[self.date]\
                                  * (1 - self.frac_contact_active) \
                                  * results['antibody_coverage'] * self.antibody_sensitivity
        # assuming cases are detected and quarantined in the middle of infectious period
        # and spend half of their contacts at the site.
        results['prevented_secondary_transmission'] = \
            0.5 * self.N * self.Rt.loc[self.date] * results['delta_p_infected'] * self.relative_contact_rate / 2
        results['prevented_secondary_transmission'] *= (1 + results['delta_immunity'])
        results['delta_covid_index'] = results['prevented_secondary_transmission'] \
                                     * self.projection['IHR__per_capita'].loc[self.date]
        results['test_cost_pcr'] = self.N * results['pcr_coverage'] * self.pcr_cost * results['pcr_frequency']

        outbreak_prob_no_testing = self._calculate_outbreak_prob(reduced_case_num=0)
        results['outbreak_prob'] = results.apply(
            lambda r: self._calculate_outbreak_prob(reduced_case_num=r.delta_p_infected * self.N
                                                  + r.prevented_secondary_transmission), axis=1)
        results['delta_outbreak_prob'] = outbreak_prob_no_testing - results['outbreak_prob']
        results['delta_closure_cost'] = results['delta_outbreak_prob'] * self.site_closure_cost
        results['delta_hospitalization_cost'] = results['delta_covid_index'] * self.N * self.hospitalization_cost

        results['avoided_cost'] = results['delta_closure_cost'] + results['delta_hospitalization_cost']

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

        if by == 'net_cost':
            results = self.results.iloc[np.argmin(self.results[by])].to_dict()
        elif by == 'delta_p_infected':
            results = self.results.iloc[np.argmax(self.results[by])].to_dict()
        elif by == 'outbreak_prob':
            results = self.results.iloc[np.argmin(self.results[by])].to_dict()

        return results

    def reformat_optimization_results(self, results):
        testing_parameters = ['antibody_coverage', 'pcr_coverage', 'pcr_frequency']
        params = {k: v for k, v in results.items() if k in testing_parameters}
        metrics = {k: v for k, v in results.items() if k not in testing_parameters}

        params = pd.DataFrame(params, index=['optimal'])
        metrics = pd.DataFrame(metrics, index=['metrics'])

        params = params.rename(columns={col: col + ' per month' for col in params.columns if 'pcr' in col}).T
        metrics = metrics.rename(columns={col: col + ' per month' for col in metrics.columns}).T

        return params, metrics




