#!/usr/bin/env python
"""
Run Covid data model.
Execution:
    ./run.py run-model [JHU|CDS] [--output-dir]
"""
import logging
import simplejson
import click

from libs import build_params
from libs.CovidTimeseriesModel import CovidTimeseriesModel
from libs.CovidDatasets import CDSDataset
from libs.CovidDatasets import JHUDataset

_logger = logging.getLogger(__name__)


def record_results(res, directory, name, num, pop):
    import copy
    import os.path
    vals = copy.copy(res)
    # Format the date in the manner the front-end expects
    vals['Date'] = res['Date'].apply(lambda d: f"{d.month}/{d.day}/{d.year}")
    # Set the population
    vals['Population'] = pop
    # Write the results to the specified directory
    output_file = os.path.join(
        directory, name.upper() + '.' + str(num) + '.json'
    ).format(name)
    with open(output_file, 'w') as out:
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
    hospitalization_rate = .0727
    hospitalized_cases_requiring_icu_care = .1397
    total_infected_period = 12
    model_interval = 4
    case_fatality_rate = .0109341104294479
    # TODO(chris): This overwrites the r0 from datas_sources.  Which do we want?
    r0 = 2.4

    state_population = dataset.get_population_by_country_state(country, state)
    state_timeseries = dataset.get_timeseries_by_country_state(
        country, state, model_interval
    )
    state_beds = dataset.get_beds_by_country_state(country, state)

    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    MODEL_PARAMETERS = {
        # Pack the changeable model parameters
        'timeseries': state_timeseries,
        'beds': state_beds,
        'population': state_population,
        # Number of iterations into the future to project
        'projection_iterations': 24,
        'r0': r0,
        'interventions': interventions,
        'hospitalization_rate': hospitalization_rate,
        'initial_hospitalization_rate': .05,
        'case_fatality_rate': case_fatality_rate,
        'hospitalized_cases_requiring_icu_care': hospitalized_cases_requiring_icu_care,
        # Assumes that anyone who needs ICU care and doesn't get it dies
        'case_fatality_rate_hospitals_overwhelmed': hospitalization_rate * hospitalized_cases_requiring_icu_care,
        'hospital_capacity_change_daily_rate': 1.05,
        'max_hospital_capacity_factor': 2.07,
        'initial_hospital_bed_utilization': .6,
        'model_interval': model_interval, # In days
        'total_infected_period': 12, # In days
        'rolling_intervals_for_current_infected': int(round(total_infected_period / model_interval, 0)),
    }
    return CovidTimeseriesModel().forecast(model_parameters=MODEL_PARAMETERS)


@click.group()
def main():
    """Entrypoint for Data Model CLI."""
    pass


@main.command()
@click.argument('dataset_name', type=click.Choice(['JHU', 'CDS']), default="CDS")
@click.option(
    '--output-dir', default='results/test', help="Model results output directory"
)
@click.option('--country', default='USA')
@click.option('--state', '-s', multiple=True)
def run_model(dataset_name, output_dir, country, state):
    # filter_past_date = datetime.date(2020, 3, 19)
    filter_past_date = None
    if dataset_name == 'JHU':
        dataset = JHUDataset(filter_past_date=filter_past_date)
    elif dataset_name == 'CDS':
        dataset = CDSDataset(filter_past_date=filter_past_date)

    # `state` is always a list (but the naming doesn't make it plural)
    # If no states are provided, get all states for a given country.
    states = state or dataset.get_all_states_by_country(country)
    _logger.info(
        f'Running model on {dataset.__class__.__name__}. Saving output to {output_dir}'
    )

    for state in states:
        for i, intervention in enumerate(build_params.INTERVENTIONS):
            _logger.info(f"Running intervention {i} for {state}")
            model_output = model_state(dataset, 'USA', state, intervention)
            record_results(
                model_output,
                output_dir,
                state,
                i,
                dataset.get_population_by_country_state('USA', state)
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
