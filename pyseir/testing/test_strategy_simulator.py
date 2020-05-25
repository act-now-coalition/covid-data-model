import numpy as np
import pandas as pd
from enum import Enum
from pyseir.testing.load_data import load_population_size, load_rt, load_projection
from sklearn.model_selection import ParameterGrid
from datetime import datetime, date, timedelta


PCR_COST = 250
PCR_SENSITIVITY = 0.7
ANTIBODY_COST = 120
ANTIBODY_SENSITIVITY = 0.5
ANTIBODY_FALSE_POSITIVITY = 0.1
MAX_PCR_AVAILABILITY = 300
MAX_ANTIBODY_AVAILABILITY = 1000
MIN_NUM_CASE_OUTBREAK = 3
HOSPITALIZATION_COST = 3000  # this maybe way off the real number
MAX_TEST_COST_PER_MONTH = 100000
DATE = datetime.today()
SITE_CLOSURE_COST = 200000
DATE_FORMAT = "%Y-%m-%d"

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
                 pcr_frequency=[1/7, 1/14, 1/30],
                 relative_contact_rate=1,
                 frac_contact_active=1,
                 min_num_case_outbreak=MIN_NUM_CASE_OUTBREAK,
                 pcr_max_availability_per_month=MAX_PCR_AVAILABILITY,
                 antibody_max_availability=MAX_ANTIBODY_AVAILABILITY,
                 max_test_cost_per_month=MAX_TEST_COST_PER_MONTH,
                 num_days_aggregate = 30,
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

        self.max_test_cost_per_month = max_test_cost_per_month

        self.num_days_aggregate = num_days_aggregate

        self.results = None


    def _index_time_to_str(self, df):
        """

        """
        df.index = df.index.strftime(DATE_FORMAT)
        return df

    def run(self):
        """

        """
        param_grid = {'pcr_coverage': self.pcr_coverage,
                      'antibody_coverage': self.antibody_coverage,
                      'pcr_frequency': self.pcr_frequency}

        results = pd.DataFrame(list(ParameterGrid(param_grid)))
        results['delta_p_infected'] = (self.projection['I'] / self.projection['N']).loc[self.date] * \
                                       results['pcr_coverage'] * self.pcr_sensitivity * results['pcr_frequency'] * \
                                       self.num_days_aggregate
        results['delta_immunity'] = (self.projection['R'] / self.projection['N']).loc[self.date]\
                                  * (1 - self.frac_contact_active) \
                                  * results['antibody_coverage'] * self.antibody_sensitivity

        # assuming cases are detected and quarantined in the middle of infectious period
        results['prevented_secondary_transmission'] = \
            self.N * self.Rt.loc[self.date] * results['delta_p_infected'] * self.relative_contact_rate / 2
        results['prevented_secondary_transmission'] *= (1 + results['delta_immunity'])
        results['delta_covid_index'] = results['prevented_secondary_transmission'] \
                                     * self.projection['IHR__per_capita'].loc[self.date]
        results['test_cost_pcr'] = self.N * results['pcr_coverage'] * self.pcr_cost
        results['test_cost_ab'] = self.N * results['antibody_coverage'] * self.antibody_cost

        # assuming antibody test is given only once
        results['test_cost'] = results['test_cost_pcr'] * results['pcr_frequency'] * self.num_days_aggregate \
                             + results['test_cost_ab']

        # reduction in outbreak probability as reduction in ratio of reduced number of infections over
        # minimum number of cases that trigger an outbreak
        results['delta_outbreak_prob'] = np.minimum(1, (results['delta_p_infected'] * self.N + results[
            'prevented_secondary_transmission']) / self.min_num_case_outbreak)

        results['avoided_cost'] = results['delta_covid_index'] * self.N * self.hospitalization_cost \
                                + results['delta_outbreak_prob'] * self.site_closure_cost

        results['hospitalization_cost_no_test'] = self.N * (self.projection['I'] * self.projection['IHR__per_capita'] /
                                                  self.projection['N']).loc[self.date] * self.hospitalization_cost
        results['worksite_outbreak_cost_no_test'] = np.minimum(1, self.N * (np.append([0],
                                                                                      np.diff(self.projection['I']))/
                                                                         self.projection['N']).loc[self.date] *
                                                               30 / self.min_num_case_outbreak) * self.site_closure_cost
        results['cost_no_test'] = results['hospitalization_cost_no_test'] + results['worksite_outbreak_cost_no_test']
        results['net_cost'] = results['cost_no_test'] + results['test_cost'] - results['avoided_cost']

        under_pcr_max_capacity = self.N * results['pcr_coverage'] * results['pcr_frequency'] * 30 \
                                 <= self.pcr_max_availability_per_month
        under_ab_max_capacity = self.N * results['antibody_coverage'] <= self.antibody_max_availability
        under_max_test_cost = results['test_cost'] * 30 / self.num_days_aggregate <= self.max_test_cost_per_month

        self.results = results[under_pcr_max_capacity & under_ab_max_capacity & under_max_test_cost]

        return self.results


    def optimize(self, by='net_cost'):
        if self.results is None:
            self.results = self.run()

        if by == 'net_cost':
            return self.results.iloc[np.argmin(self.results[by])].to_dict()
        elif by == 'delta_p_infected':
            return self.results.iloc[np.argmax(self.results[by])].to_dict()
        elif by == 'delta_outbreak_p':
            return self.results.iloc[np.argmax(self.results[by])].to_dict()

