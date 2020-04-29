import pandas as pd
import numpy as np
from enum import Enum
from pyseir import load_data

class CovidMeasure(Enum):
    HOSPITALIZATION_GENERAL_INFECTED = 'hospitalization_general_infected'
    HOSPITALIZATION_ICU_INFECTED = 'hospitalization_icu_infected'
    HOSPITALIZATION_INFECTED = 'hospitalization_infected'
    MORTALITY_INFECTED = 'mortality_infected'
    HOSPITALIZATION_GENERAL = 'hospitalization_general'
    HOSPITALIZATION_ICU = 'hospitalization_icu'
    HOSPITALIZATION = 'hospitalization'
    MORTALITY = 'mortality'

def reconstruct_age_specific_mortality(seir_model):
    """
    Reconstruct age specific mortality given
    """


class DemographicMapper:
    """
    R
    """

    def __init__(self,
                 fips,
                 mle_model,
                 fit_results,
                 start_date=None,
                 end_date=None):

        self.fips = fips
        self.seir_predictions = seir_predictions
        self.target_variables = target_variables
        self.predicted_target_variables = \
            self.predictiton_to_target_variables(target_variables)
        self.prediction_time = prediction_time
        self.prediction_age_groups = prediction_age_groups
        if customer_age_distribution is None:
            customer_age_distribution = \
                self.generate_customer_age_distribution()
        self.customer_age_distribution = customer_age_distribution
        self.fips_age_distribution = fips_age_distribution
        self.aggregated_predictions = None
        self.risk_ratio_by_age_group = risk_ratio_by_age_group



    def generate_customer_age_distribution(self):
        """
        Generates age distribution function.
        """
        return

    def calculate_predicted_total_infections(self, seir_predictions=None):
        """

        """
        if seir_predictions is None:
            seir_predictions = self.seir_predictions

        total_infected = np.zeros(len(self.prediction_age_groups))
        for c in ['E', 'I', 'A', 'HGen', 'HICU', 'HVent']:
            total_infected += seir_predictions[c]
        total_infected += seir_predictions['R'] / len(self.prediction_age_groups)
        return total_infected


    def prediction_to_target_variables(self, seir_predictions=None,
                                       target_variables=None):
        """

        """

        target_variable_trajectories = dict()

        total_infected = self.calculate_predicted_total_infections(self.seir_predictions)

        for var_name in target_variables:
            var = CovidMeasure(var_name)
            if var is CovidMeasure.HOSPITALIZATION_GENERAL_INFECTED:
                target_variables[var_name] = seir_predictions['HGen'] / total_infected

            elif var is CovidMeasure.HOSPITALIZATION_GENERAL:
                target_variables[var_name] = seir_predictions['HGen'] / self.fips_age_distribution

            elif var is CovidMeasure.HOSPITALIZATION_ICU_INFECTED:
                target_variables[var_name] = seir_predictions['HICU'] + seir_predictions['HVent'] / total_infected

            elif var is CovidMeasure.HOSPITALIZATION_ICU:
                target_variables[var_name] = seir_predictions['HICU'] + seir_predictions['HVent'] / self.fips_age_distribution

            elif var is CovidMeasure.HOSPITALIZATION_INFECTED:
                target_variables[var_name] = seir_predictions['HGen'] \
                                           + seir_predictions['HICU'] \
                                           + seir_predictions['HVent'] / total_infected

            elif var is CovidMeasure.HOSPITALIZATION:
                target_variables[var_name] = \
                    seir_predictions['HICU'] + seir_predictions['HVent'] / self.fips_age_distribution

            elif var is CovidMeasure.MORTALITY_INFECTED:
                target_variables[var_name] =


        for key in target_variable_trajectories:
            target_variable_trajectories[key] = \
                pd.DataFrame(target_variable_trajectories[key],
                             columns=self.prediction_age_groups,
                             index=pd.DatetimeIndex(self.prediction_time))




    def generate_aggregate_predictions(self):
        # calculate weights
        weights = self.age_distribution(self.prediction_age_groups)
        if self.risk_ratio_by_age_group is not None:
            weights *= self.risk_ratio_by_age_group
        # aggregate predictions
        aggregated_predictions = self.predicted_target_variables.dot(weights)
        return aggregated_predictions

