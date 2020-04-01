import logging
import time
import datetime
import pathlib
import simplejson
import json
from collections import defaultdict
from libs.CovidDatasets import CDSDataset, JHUDataset
from libs.CovidTimeseriesModelSIR import CovidTimeseriesModelSIR
from libs.build_params import r0, OUTPUT_DIR, INTERVENTIONS
import os.path
import pandas as pd

# from libs.CovidDatasets import CDSDataset, JHUDataset
from libs.CovidTimeseriesModelSIR import CovidTimeseriesModelSIR
from libs.datasets import JHUDataset
from libs.datasets import FIPSPopulation
from libs.datasets import CDSDataset
from libs.datasets import DHBeds
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.build_params import r0, OUTPUT_DIR, INTERVENTIONS

_logger = logging.getLogger(__name__)


def prepare_data_for_website(data, population, min_begin_date, max_end_date, interval: int = 4):
    """Prepares data for website output."""
    # Indexes used by website JSON:
    # date: 0,
    # hospitalizations: 8,
    # cumulativeInfected: 9,
    # cumulativeDeaths: 10,
    # beds: 11,
    # totalPopulation: 16,

    # Columns from Harvard model output:
    # date, total, susceptible, exposed, infected, infected_a, infected_b, infected_c, recovered, dead
    # infected_b == Hospitalized
    # infected_c == Hospitalized in ICU

    cols = [
        "date",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "infected_b",
        "infected",
        "dead",
        "beds",
        "i",
        "j",
        "k",
        "l",
        "population",
        "m",
        "n",
    ]

    website_ordering = pd.DataFrame(data, columns=cols).fillna(0)

    # @TODO: Find a better way of restricting to every fourth day.
    #        Alternatively, change the website's expectations.
    website_ordering = website_ordering[website_ordering.index % interval == 0].reset_index()

    if min_begin_date:
        website_ordering = pd.DataFrame(
            website_ordering[website_ordering["date"] >= min_begin_date]
        )
    if max_end_date:
        website_ordering = pd.DataFrame(
            website_ordering[website_ordering["date"] <= max_end_date]
        )

    website_ordering["date"] = website_ordering["date"].dt.strftime("%-m/%-d/%y")
    website_ordering["population"] = population
    website_ordering = website_ordering.astype(
        {
            "infected_b": int,
            "infected": int,
            "dead": int,
            "beds": int,
            "population": int,
        }
    )
    website_ordering = website_ordering.astype(
        {
            "infected_b": str,
            "infected": str,
            "dead": str,
            "beds": str,
            "population": str,
        }
    )
    return website_ordering


def write_results(data, directory, name):
    """Write dataset results.

    Args:
        data: Dataframe to write.
        directory: base output directory.
        path: Name of file.
    """
    path = os.path.join(directory, name)
    with open(path, "w") as out:
        simplejson.dump(data.values.tolist(), out, ignore_nan=True)


def model_state(timeseries, population, starting_beds, interventions=None):

    # Constants
    start_time = time.time()
    HOSPITALIZATION_RATE = 0.0727
    HOSPITALIZED_CASES_REQUIRING_ICU_CARE = 0.1397
    TOTAL_INFECTED_PERIOD = 12
    MODEL_INTERVAL = 4
    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    MODEL_PARAMETERS = {
        # Pack the changeable model parameters
        "timeseries": timeseries,
        "beds": starting_beds,
        "population": population,
        # 'projection_iterations': 25, # Number of iterations into the future to project
        "projection_iterations": 80,  # Number of iterations into the future to project
        "r0": r0,
        "interventions": interventions,
        "hospitalization_rate": HOSPITALIZATION_RATE,
        "case_fatality_rate": 0.0109341104294479,
        "hospitalized_cases_requiring_icu_care": HOSPITALIZED_CASES_REQUIRING_ICU_CARE,
        # Assumes that anyone who needs ICU care and doesn't get it dies
        "case_fatality_rate_hospitals_overwhelmed": HOSPITALIZATION_RATE
        * HOSPITALIZED_CASES_REQUIRING_ICU_CARE,
        "hospital_capacity_change_daily_rate": 1.01227,  # Equivalent to 1.05 per four days
        "max_hospital_capacity_factor": 2.07,
        "initial_hospital_bed_utilization": 0.6,
        "model_interval": 4,  # In days
        "total_infected_period": 12,  # In days
        "rolling_intervals_for_current_infected": int(
            round(TOTAL_INFECTED_PERIOD / MODEL_INTERVAL, 0)
        ),
        "estimated_new_cases_per_death": 32,
        "estimated_new_cases_per_confirmed": 20,
        # added for seird model
        "incubation_period": 5,  # In days
        "duration_mild_infections": 10,  # In days
        "icu_time_death": 7,  # Time from ICU admission to death, In days
        "hospital_time_recovery": 11,  # Duration of hospitalization, In days
        # If True use the harvard parameters directly, if not calculate off the above
        "use_harvard_params": True,
        # If True use the parameters that make R0 2.4, if not calculate off the above
        "fix_r0": False,
        # If True use the harvard model inputs for inital conditions and N (recreate their graph)
        "use_harvard_init": False,
        "use_harvard_params": False,  # If True use the harvard parameters directly, if not calculate off the above
        "fix_r0": False,  # If True use the parameters that make R0 2.4, if not calculate off the above
        "hospitalization_rate": HOSPITALIZATION_RATE,
        "hospitalized_cases_requiring_icu_care": HOSPITALIZED_CASES_REQUIRING_ICU_CARE,
        "total_infected_period": 12,  # In days
        "duration_mild_infections": 6,  # In days
        "hospital_time_recovery": 11,  # Duration of hospitalization, In days
        "icu_time_death": 7,  # Time from ICU admission to death, In days
        "case_fatality_rate": 0.0109341104294479,
        "beta": 0.5,
        "beta_hospitalized": 0.1,
        "beta_icu": 0.1,
        "presymptomatic_period": 3,
        "exposed_from_infected": True,
        #'model': 'sir',
        "model": "seir",
    }
    MODEL_PARAMETERS["exposed_infected_ratio"] = 1 / MODEL_PARAMETERS["beta"]

    [results, soln] = CovidTimeseriesModelSIR().forecast_region(
        model_parameters=MODEL_PARAMETERS
    )

    available_beds = starting_beds * (
        1 - MODEL_PARAMETERS["initial_hospital_bed_utilization"]
    )

    results["beds"] = list(
        min(
            available_beds
            * MODEL_PARAMETERS["hospital_capacity_change_daily_rate"] ** exp,
            available_beds * MODEL_PARAMETERS["max_hospital_capacity_factor"],
        )
        for exp in range(0, len(results.index))
    )

    return results


def build_county_summary(country='USA', state=None):
    """Builds county summary json files."""
    beds_data = DHBeds.build_from_local_github().to_generic_beds()
    population_data = FIPSPopulation().to_generic_population()
    timeseries = JHUDataset.build_from_local_github().to_generic_timeseries()
    timeseries = timeseries.get_subset(
        AggregationLevel.COUNTY, after=min_date, country=country, state=state
    )

    output_dir = pathlib.Path(OUTPUT_DIR) / "county_summaries"
    _logger.info(f"Outputting to {output_dir}")
    if not output_dir.exists():
        _logger.info(f"{output_dir} does not exist, creating")
        output_dir.mkdir(parents=True)

    counties_by_state = defaultdict(list)
    for country, state, county, fips in timeseries.county_keys():
        counties_by_state[state].append((county, fips))

    for state, counties in counties_by_state.items():
        data = {
            'counties_with_data': []
        }
        for county, fips in counties:
            cases = timeseries.get_data(state=state, country=country, fips=fips)
            beds = beds_data.get_county_level(state, fips=fips)
            population = population_data.get_county_level(country, state, fips=fips)
            if population and beds and sum(cases.cases):
                data['counties_with_data'].append(fips)

        output_path = output_dir / f"{state}.summary.json"
        output_path.write_text(json.dumps(data, indent=2))


def run_county_level_forecast(min_date, max_date, country='USA', state=None):
    beds_data = DHBeds.build_from_local_github().to_generic_beds()
    population_data = FIPSPopulation().to_generic_population()
    timeseries = JHUDataset.build_from_local_github().to_generic_timeseries()
    timeseries = timeseries.get_subset(
        AggregationLevel.COUNTY, after=min_date, country=country, state=state
    )

    output_dir = pathlib.Path(OUTPUT_DIR) / "county"
    _logger.info(f"Outputting to {output_dir}")
    if not output_dir.exists():
        _logger.info(f"{output_dir} does not exist, creating")
        output_dir.mkdir(parents=True)

    counties_by_state = defaultdict(list)
    county_keys = timeseries.county_keys()
    for country, state, county, fips in county_keys:
        counties_by_state[state].append((county, fips))

    processed = 0
    skipped = 0
    total = len(county_keys)
    for state, counties in counties_by_state.items():
        _logger.info(f'Running county models for {state}')
        data = []
        for county, fips in counties:
            if (processed + skipped) % 200 == 0:
                _logger.info(f"Processed {processed + skipped} / {total} - "
                             f"Skipped {skipped} due to missing data")

            _logger.debug(f'Running model for county: {county}, {state} - {fips}')
            cases = timeseries.get_data(state=state, country=country, fips=fips)
            beds = beds_data.get_county_level(state, fips=fips)
            population = population_data.get_county_level(country, state, fips=fips)

            total_cases = sum(cases.cases)
            if not population or not beds or not total_cases:
                _logger.debug(
                    f"Missing data, skipping: Beds: {beds} Pop: {population} Total Cases: {total_cases}"
                )
                skipped += 1
                continue
            else:
                processed += 1

            for i, intervention in enumerate(INTERVENTIONS):
                _logger.debug(
                    f"Running intervention {i} for {state} - "
                    f"total cases: {total_cases} beds: {beds} pop: {population}"
                )
                results = model_state(cases, beds, population, intervention)
                website_data = prepare_data_for_website(results, population, min_date, max_date, interval=4)
                write_results(website_data, output_dir, '{state}.{fips}.{i}.json')


def run_state_level_forecast(min_date, max_date, country='USA', state=None):
    beds_data = DHBeds.build_from_local_github().to_generic_beds()
    population_data = FIPSPopulation().to_generic_population()
    timeseries = JHUDataset.build_from_local_github().to_generic_timeseries()
    timeseries = timeseries.get_subset(
        AggregationLevel.STATE, after=min_date, country=country, state=state
    )
    output_dir = pathlib.Path(OUTPUT_DIR) / "state"
    _logger.info(f"Outputting to {output_dir}")
    if not output_dir.exists():
        _logger.info(f"{output_dir} does not exist, creating")
        output_dir.mkdir(parents=True)

    for state in timeseries.states:
        _logger.info(f'Generating data for state: {state}')
        cases = timeseries.get_data(state=state)
        beds = beds_data.get_state_level(state)
        population = population_data.get_state_level(country, state)
        if not population:
            _logger.warning(f"Missing population for {state}")
            continue

        for i, intervention in enumerate(INTERVENTIONS):
            _logger.info(f"Running intervention {i} for {state}")
            results = model_state(cases, beds, population, intervention)
            website_data = prepare_data_for_website(results, population, min_date, max_date, interval=4)
            write_results(website_data, OUTPUT_DIR, '{state}.{i}.json')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # @TODO: Remove interventions override once support is in the Harvard model.
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)
    # build_counties_with_data()
    run_county_level_forecast(min_date, max_date)
    # run_state_level_forecast(min_date, max_date)
