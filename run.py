#!/usr/bin/env python
"""
Run Covid data model.

Execution:
    ./run.py run-model

"""
import logging


import datetime
import time
import simplejson
import click

from libs.CovidTimeseriesModel import CovidTimeseriesModel
from libs.CovidDatasets import CDSDataset

_logger = logging.getLogger(__name__)


def record_results(result, directory, name, num, pop):
    import copy
    import os.path
    vals = copy.copy(result)
    # Format the date in the manner the front-end expects
    vals['Date'] = result['Date'].apply(lambda d: "{}/{}/{}".format(d.month, d.day, d.year))
    # Set the population
    vals['Population'] = pop
    # Write the results to the specified directory
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

def model_state(country, state, interventions=None):
    ## Constants
    start_time = time.time()
    HOSPITALIZATION_RATE = .0727
    HOSPITALIZED_CASES_REQUIRING_ICU_CARE = .1397
    TOTAL_INFECTED_PERIOD = 12
    MODEL_INTERVAL = 4
    r0 = 2.4
    Dataset = CDSDataset(filter_past_date=datetime.date(2020, 3, 19))
    POP = Dataset.get_population_by_country_state(country, state)
    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    MODEL_PARAMETERS = {
        # Pack the changeable model parameters
        'timeseries': Dataset.get_timeseries_by_country_state(country, state, MODEL_INTERVAL),
        'beds': Dataset.get_beds_by_country_state(country, state),
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

r0 = 2.4

INTERVENTIONS = [
    None,  # No Intervention
    {  # Flatten the Curve
        datetime.date(2020, 3, 23): 1.3,
        datetime.date(2020, 4, 20): 1.1,
        datetime.date(2020, 5, 22): 0.8,
        datetime.date(2020, 6, 23): r0
    },
    {  # Full Containment
        datetime.date(2020, 3, 23): 1.3,
        datetime.date(2020, 3, 31): 0.3,
        datetime.date(2020, 4, 28): 0.2,
        datetime.date(2020, 5,  6): 0.1,
        datetime.date(2020, 5, 10): 0.35,
        datetime.date(2020, 5, 18): r0
    },
    {  # @TODO: Model w/ FlatteningTheCurve (2 wk delay)
        datetime.date(2020, 3, 23): 1.3,
        datetime.date(2020, 4, 20): 1.1,
        datetime.date(2020, 5, 22): 0.8,
        datetime.date(2020, 6, 23): r0
    },
    {  # @TODO: Model w/ FlatteningTheCurve (1 mo delay)
        datetime.date(2020, 3, 23): 1.3,
        datetime.date(2020, 4, 20): 1.1,
        datetime.date(2020, 5, 22): 0.8,
        datetime.date(2020, 6, 23): r0
    },
    {  # @TODO: Full Containment (1 wk dly)
        datetime.date(2020, 3, 23): 1.3,
        datetime.date(2020, 3, 31): 0.3,
        datetime.date(2020, 4, 28): 0.2,
        datetime.date(2020, 5,  6): 0.1,
        datetime.date(2020, 5, 10): 0.35,
        datetime.date(2020, 5, 18): r0
    },
    {  # @TODO: Full Containment (2 wk dly)
        datetime.date(2020, 3, 23): 1.3,
        datetime.date(2020, 3, 31): 0.3,
        datetime.date(2020, 4, 28): 0.2,
        datetime.date(2020, 5,  6): 0.1,
        datetime.date(2020, 5, 10): 0.35,
        datetime.date(2020, 5, 18): r0
    },
    {  # Social Distancing
        datetime.date(2020, 3, 23): 1.7,
        datetime.date(2020, 6, 23): r0
    },
]


@click.group()
def main():
    pass


@main.command()
@click.option(
    '--output-dir', default='results/test', help="Model results output directory"
)
def run_model(output_dir):

    dataset = CDSDataset()
    logging.info(f'Running model on {dataset.__class__.__name__}. Saving output to {output_dir}')
    for state in dataset.get_all_states_by_country('USA'):
        for i in range(0, len(INTERVENTIONS)):
            _logger.info(f"Running intervention {i} for {state}")
            intervention = INTERVENTIONS[i]
            record_results(
                model_state('USA', state, intervention),
                output_dir,
                state,
                i,
                dataset.get_population_by_country_state('USA', state)
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
