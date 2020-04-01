import logging

import datetime
import time
import simplejson
from libs.build_params import OUTPUT_DIR, get_interventions
from libs.CovidTimeseriesModel import CovidTimeseriesModel
from libs.CovidDatasets import CDSDataset

_logger = logging.getLogger(__name__)


def record_results(res, directory, name, num, pop):
    import copy
    import os.path
    vals = copy.copy(res)
    # Format the date in the manner the front-end expects
    vals['Date'] = res['Date'].apply(lambda d: "{}/{}/{}".format(d.month, d.day, d.year))
    # Set the population
    vals['Population'] = pop
    # Write the results to the specified directory
    os.makedirs(directory, exist_ok=True)
    with open( os.path.join(directory, name.upper() + '.' + str(num) + '.json').format(name), 'w') as out:
        simplejson.dump(vals[[
                'Date',
                'R',
                'Beg. Susceptible',
                'New Inf.',
                'Curr. Inf.',
                'Recov. or Died',
                'End Susceptible',
                'Actual Reported',
                'Pred. Hosp.',
                'Cum. Inf.',
                'Cum. Deaths',
                'Avail. Hosp. Beds',
                'S&P 500',
                'Est. Actual Chance of Inf.',
                'Pred. Chance of Inf.',
                'Cum. Pred. Chance of Inf.',
                'Population',
                'R0',
                '% Susceptible'
            ]].values.tolist(), out, ignore_nan=True)

def model_state(dataset, country, state, interventions=None):
    ## Constants
    start_time = time.time()
    HOSPITALIZATION_RATE = .0727
    HOSPITALIZED_CASES_REQUIRING_ICU_CARE = .1397
    TOTAL_INFECTED_PERIOD = 12
    MODEL_INTERVAL = 4
    r0 = 2.4
    POP = dataset.get_population_by_country_state(country, state)
    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    MODEL_PARAMETERS = {
        # Pack the changeable model parameters
        'timeseries': dataset.get_timeseries_by_country_state(country, state, MODEL_INTERVAL),
        'beds': dataset.get_beds_by_country_state(country, state),
        'population': POP,
        'projection_iterations': 24, # Number of iterations into the future to project
        'r0': r0,
        'interventions': interventions,
        'hospitalization_rate': HOSPITALIZATION_RATE,
        'initial_hospitalization_rate': .05,
        'case_fatality_rate': .0109341104294479,
        'hospitalized_cases_requiring_icu_care': HOSPITALIZED_CASES_REQUIRING_ICU_CARE,
        # Assumes that anyone who needs ICU care and doesn't get it dies
        'case_fatality_rate_hospitals_overwhelmed': HOSPITALIZATION_RATE * HOSPITALIZED_CASES_REQUIRING_ICU_CARE,
        'hospital_capacity_change_daily_rate': 1.05,
        'max_hospital_capacity_factor': 2.07,
        'initial_hospital_bed_utilization': .6,
        'model_interval': 4, # In days
        'total_infected_period': 12, # In days
        'rolling_intervals_for_current_infected': int(round(TOTAL_INFECTED_PERIOD / MODEL_INTERVAL, 0)),
    }
    return CovidTimeseriesModel().forecast(model_parameters=MODEL_PARAMETERS)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset = CDSDataset()
    interventions = get_interventions()
    for state in dataset.get_all_states_by_country('USA'):
        for i, intervention in enumerate(interventions):
            _logger.info(f"Running intervention {i} for {state}")
            record_results(
                model_state(dataset, 'USA', state, intervention),
                OUTPUT_DIR,
                state,
                i,
                dataset.get_population_by_country_state('USA', state)
            )
