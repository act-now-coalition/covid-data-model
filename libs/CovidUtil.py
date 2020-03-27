from libs.CovidDatasets import CDSDataset
from libs.CovidTimeseriesModel import CovidTimeseriesModel


class CovidUtil:
    # Class for holding the tools and logic that don't really fit anywhere else (yet)
    def initialize_model_parameters(self, model_params):
        #  Some of the model parameters are interdependent, which can make them clumsy to change around. This
        #  function is intended to hold all of the logic for computing the necessary derivative model parameters
        model_params['case_fatality_rate_hospitals_overwhelmed'] = \
            model_params['hospitalization_rate'] * model_params['hospitalized_cases_requiring_icu_care']
        model_params['rolling_intervals_for_current_infected'] = \
            int(round(model_params['total_infected_period'] / model_params['model_interval'], 0))
        return model_params

    def model_us_state(self, state, dataset, model_interval, interventions=None):
        ## Constants
        # Pack all of the assumptions and parameters into a dict that can be passed into the model
        MODEL_PARAMETERS = {
            # Pack the changeable model parameters
            'timeseries': dataset.get_timeseries_by_country_state('USA', state, model_interval),
            'beds': dataset.get_beds_by_country_state('USA', state),
            'population': dataset.get_population_by_country_state('USA', state),
            'projection_iterations': 25,  # Number of iterations into the future to project
            'r0': 2.4,
            'interventions': interventions,
            'hospitalization_rate': .0727,
            'initial_hospitalization_rate': .05,
            'case_fatality_rate': .0109341104294479,
            'hospitalized_cases_requiring_icu_care': .1397,
            # Assumes that anyone who needs ICU care and doesn't get it dies
            'hospital_capacity_change_daily_rate': 1.05,
            'max_hospital_capacity_factor': 2.07,
            'initial_hospital_bed_utilization': .6,
            'model_interval': 4,  # In days
            'total_infected_period': 12,  # In days
        }
        MODEL_PARAMETERS = self.initialize_model_parameters(MODEL_PARAMETERS)
        return CovidTimeseriesModel().forecast(model_parameters=MODEL_PARAMETERS)
