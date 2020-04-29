import logging
import pandas as pd
import numpy as np
from copy import copy
from enum import Enum
from pyseir import load_data
from datetime import datetime, timedelta

class CovidMeasure(Enum):
    HOSPITALIZATION_GENERAL_INFECTED = 'hospitalization_general_infected'
    HOSPITALIZATION_ICU_INFECTED = 'hospitalization_icu_infected'
    HOSPITALIZATION_INFECTED = 'hospitalization_infected'
    MORTALITY_INFECTED = 'mortality_infected'
    HOSPITALIZATION_GENERAL = 'hospitalization_general'
    HOSPITALIZATION_ICU = 'hospitalization_icu'
    HOSPITALIZATION = 'hospitalization'
    MORTALITY = 'mortality'


class DemographicMapper:
    """
    """

    def __init__(self,
                 fips,
                 mle_model,
                 fit_results,
                 target_variables,
                 aggregate=False,
                 target_age_distribution=None,
                 risk_modifier_by_age_group=None,
                 start_date=None,
                 end_date=None):

        self.fips = fips
        self.predictions = {k: v for k, v in mle_model.results.items() if k != 'by_age'}
        self.predictions_by_age = mle_model.results['by_age']
        self.parameters = {k: v for k, v in mle_model.__dict__.items() if k not in ('by_age', 'results')}
        self.fit_results = fit_results
        self.aggregate = aggregate
        self.target_variables = target_variables
        if target_age_distribution:
            target_age_distribution = lambda x: 1
        self.target_age_distribution = target_age_distribution
        self.risk_modifier_by_age_group = risk_modifier_by_age_group

    def generate_customer_age_distribution(self):
        """
        Generates age distribution function.
        """
        return

    def reconstruct_age_specific_mortality(self):
        """
        Reconstruct age specific mortality given
        """

        # calculate the derivatives
        mortality_rate_ICU = np.tile(self.parameters['mortality_rate_from_ICU'],
                                     (self.parameters['t_list'].shape[0], 1)).T
        mortality_rate_ICU[:, np.where(self.predictions['HICU'] > self.parameters['beds_ICU'])] = \
            self.parameters['mortality_rate_no_ICU_beds']

        mortality_rate_NonICU = np.tile(self.parameters['mortality_rate_from_hospital'],
                                        (self.parameters['t_list'].shape[0], 1)).T
        mortality_rate_NonICU[:, np.where(self.predictions['HGen'] > self.parameters['beds_general'])] = \
            self.parameters['mortality_rate_no_general_beds']

        died_from_hosp = self.predictions_by_age['HGen'] * mortality_rate_NonICU \
                         / self.parameters['hospitalization_length_of_stay_general']
        died_from_icu = self.predictions_by_age['HICU'] * (
                1 - self.parameters['fraction_icu_requiring_ventilator']) * self.parameters['mortality_rate_ICU'] / \
                        self.parameters['hospitalization_length_of_stay_icu']
        died_from_icu_vent = self.predictions_by_age['HVent'] * self.parameters['mortality_rate_from_ICUVent'] / \
                             self.parameters['hospitalization_length_of_stay_icu_and_ventilator']
        dDdt = died_from_hosp + died_from_icu + died_from_icu_vent
        delta_t = np.diff(self.parameters['t_list'])
        delta_t = np.append(delta_t, delta_t[-1])
        dD = dDdt * delta_t[:, np.newaxis].T
        mortality = np.cumsum(dD, axis=1)
        return mortality

    def calculate_predicted_total_infections(self):
        """

        """

        total_infected = np.zeros(len(self.parameters['age_groups']))
        for c in ['E', 'I', 'A', 'HGen', 'HICU', 'HVent']:
            total_infected += self.predictions_by_age[c]
        return total_infected


    def generate_target_predictions(self, target_variables=None):
        """

        """

        target_variables = target_variables or self.target_variables

        total_infected = self.calculate_predicted_total_infections()

        mortality = self.reconstruct_age_specific_mortality()

        target_predictions = dict()

        for var_name in target_variables:
            var = CovidMeasure(var_name)
            if var is CovidMeasure.HOSPITALIZATION_GENERAL_INFECTED:
                target_predictions[var_name] = self.predictions_by_age['HGen'] / total_infected

            elif var is CovidMeasure.HOSPITALIZATION_GENERAL:
                target_predictions[var_name] = self.predictions_by_age['HGen'] / self.parameters['N']

            elif var is CovidMeasure.HOSPITALIZATION_ICU_INFECTED:
                target_predictions[var_name] = \
                    self.predictions_by_age['HICU'] + self.predictions_by_age['HVent'] / total_infected

            elif var is CovidMeasure.HOSPITALIZATION_ICU:
                target_predictions[var_name] = self.predictions_by_age['HICU'] \
                                           + self.predictions_by_age['HVent'] / self.parameters['N']

            elif var is CovidMeasure.HOSPITALIZATION_INFECTED:
                target_predictions[var_name] = \
                    self.predictions_by_age['HGen'] \
                  + self.predictions_by_age['HICU']\
                  + self.predictions_by_age['HVent'] / total_infected

            elif var is CovidMeasure.HOSPITALIZATION:
                target_predictions[var_name] = \
                    self.predictions_by_age['HGen'] \
                  + self.predictions_by_age['HICU'] \
                  + self.predictions_by_age['HVent'] / self.parameters['N']

            elif var is CovidMeasure.MORTALITY_INFECTED:
                target_variables[var_name] = mortality / (total_infected + mortality)

            elif var is CovidMeasure.MORTALITY:
                target_variables[var_name] = mortality / self.parameters['N']

        for key in target_predictions:
            target_predictions[key] = \
                pd.DataFrame(target_predictions[key],
                             columns=['-'.join([str(int(tup[0])), str(int(tup[1]))]) for tup in
                                      self.parameters['age_groups']],
                             index=pd.DatetimeIndex([self.fit_results.t0 + timedelta(days=t) for t in
                                                     self.parameters['t_list']]))

        return target_predictions


    def generate_aggregated_target_predictions(self, target_predictions):
        """

        """
        # calculate weights
        age_bin_centers = [np.mean(tup) for tup in self.parameters['age_groups']]
        weights = self.target_age_distribution(age_bin_centers)
        
        if isinstance(weights, float):
            if weights == 1:
                logging.warning('no target age distribution is given, measure is aggregated assuming age '
                                'distrubtion at given FIPS code')

        # aggregate predictions
        aggregated_target_predictions = {}

        for var_name in target_predictions:
            var = CovidMeasure(var_name)
            modified_weights = weights
            if self.risk_modifier_by_age_group is not None:
                if var_name in self.risk_modifier_by_age_group:
                    modified_weights = weights * self.risk_ratio_by_age_group[var_name]
            aggregated_target_predictions[var_name] = target_predictions[var_name].dot(modified_weights)

        return aggregated_target_predictions

    def run(self):
        """
        
        """
        target_predictions = self.generate_target_predictions()
        if self.aggregate:
            aggregated_target_predictions = self.generate_aggregated_target_predictions(target_predictions)
            return aggregated_target_predictions
        else:
            return target_predictions
