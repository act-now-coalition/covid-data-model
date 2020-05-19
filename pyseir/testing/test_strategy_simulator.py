import numpy as np
import pandas as pd
from enum import Enum
from pyseir.testing.load_data import load_population_size, load_Rt, load_projection
from sklearn.model_selection import ParameterGrid
from datetime import datetime, date


PCR_COST = 250
PCR_SENSITIVITY = 0.7
ANTIBODY_COST = 120
ANTIBODY_SENSITIVITY = 0.5
ANTIBODY_FALSE_POSITIVITY = 0.1
MAX_PCR_AVAILABILITY = 100
MAX_ANTIBODY_AVAILABILITY = 500
MIN_NUM_CASE_OUTBREAK = 3
HOSPITALIZATION_COST = 5000  # this maybe way off the real number, just use a number to start with
DATE = datetime.today()


class Allocation(Enum):
    RANDOM = 'random'
    STRATIFIED = 'stratified'
    ADAPTIVE = 'adaptive'


class TestStrategySimulator:
    """
    delta P(infected) = P(infected) * cov_pcr *  sen_pcr
    **Metrics to report:**
   - delta COVID Index: reduction in COVID index:
     delta P(infected) * P(hospitalization | infected)
   - delta worksite immunity: increase in percentage of workers with immunity:
     P(recovered) * (1 - frac_worker_back) * cov_ab * sen_ab
   - prevented secondary transmission: prevented secondary transmission due to reduction in cases and increase in
   immunity:
  (N * R(t) x delta P(infection) x relative_contact_rate / 2) * (1 + delta worksite immunity) (assuming cases are     detected and quarantined in the middle of infectious period)
- testing cost:
    cov_pcr * N * cost_pcr + cov_ab * N * cost_ab
- avoided cost:
    hospitalization_cost * prevented secondary transmission * P(hospitalization | infected) + reduction in probability of outbreak(closure) * cost_of_closure
    """
    def __init__(self,
                 fips,
                 date=DATE,
                 pcr_cost=PCR_COST,
                 antibody_cost=ANTIBODY_COST,
                 pcr_sensitivity=PCR_SENSITIVITY,
                 antibody_sensitivity=ANTIBODY_SENSITIVITY,
                 antibody_false_positivity=ANTIBODY_FALSE_POSITIVITY,
                 allocation='random',
                 pcr_coverage=np.linspace(0, 1, 11),
                 antibody_coverage=np.linspace(0, 1, 11),
                 pcr_frequency=[1/7, 1/14, 1/30],
                 relative_contact_rate=1,
                 frac_contact_active=1,
                 min_num_case_outbreak=MIN_NUM_CASE_OUTBREAK,
                 pcr_max_availability_per_month=100,
                 antibody_max_availability_per_month=500,
                 max_test_cost_per_month=50000,
                 num_days_aggregate = 30
                 ):
        self.fips = fips
        self.date = date
        self.Rt = load_Rt(fips)
        self.N = load_population_size(fips)
        self.projection = load_projection(fips)
        self.relative_contact_rate = relative_contact_rate

        self.pcr_sensitivity = pcr_sensitivity
        self.pcr_cost = pcr_cost
        self.pcr_coverage = pcr_coverage
        self.pcr_frequency = pcr_frequency
        self.pcr_max_availability_per_month = pcr_max_availability_per_month

        self.antibody_cost = antibody_cost
        self.antibody_sensitivity = antibody_sensitivity
        self.antibody_coverage = antibody_coverage
        self.antibody_max_availability_per_month = antibody_max_availability_per_month
        self.antibody_false_positivity = antibody_false_positivity

        self.allocation = Allocation(allocation)
        self.frac_contact_active = frac_contact_active
        self.min_num_case_outbreak = min_num_case_outbreak

        self.max_test_cost_per_month = max_test_cost_per_month

        self.num_days_aggregate = num_days_aggregate

        self.results = None


    def run(self):
        """

        :return:
        """
        param_grid = {'pcr_coverage': self.pcr_coverage,
                      'antibody_coverage': self.antibody_coverage,
                      'pcr_frequency': self.pcr_frequency}

        results = pd.DataFrame(list(ParameterGrid(param_grid)))
        results['delta_p_infected'] = (self.projection['I'] / self.projection['N']).loc[self.date] * \
                                      results['pcr_coverage'] * self.pcr_sensitivity
        results['delta_immunity'] = (self.projection['R'] / self.projection['N']).loc[self.date]\
                                  * (1 - self.frac_contact_active) \
                                  * self.antibody_coverage * self.antibody_sensitivity
        results['delta_covid_index'] = results['delta_p_infected'] * self.projection['IHR'].loc[self.date]

        # assuming cases are detected and quarantined in the middle of infectious period
        results['prevented_secondary_transmission'] = \
            self.N * self.Rt.loc[self.date] * results['delta_p_infected'] * self.relative_contact_rate / 2
        results['prevented_secondary_transmission'] *= (1 + results['delta_immunity'])
        results['test_cost_pcr'] = self.N * self.pcr_coverage * self.pcr_cost
        results['test_cost_ab'] = self.N * self.antibody_coverage * self.antibody_costs
        results['test_cost'] = results['test_cost_pcr'] * results['pcr_frequency'] * self.num_days_aggregate \
                             + results['test_cost_ab'] * results['ab_frequency'] * self.num_days_aggregate

        under_pcr_max_capacity = self.N * results['pcr_coverage'] * results['pcr_frequency'] * 30 \
                                 <= self.pcr_max_availability_per_month
        under_ab_max_capacity = self.N * results['ab_coverage'] * results['ab_frequency'] * 30 \
                                 <= self.ab_max_availability_per_month
        under_max_test_cost = results['test_cost'] * 30 / self.num_days_aggregate <= self.max_test_cost_per_month

        self.results = results[under_pcr_max_capacity & under_ab_max_capacity & under_max_test_cost]

        return self.results



