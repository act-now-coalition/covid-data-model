from typing import Tuple, Optional, Any
import logging
import traceback
import functools
import datetime
import pathlib
import json
import os.path
from collections import defaultdict
import multiprocessing

from libs.CovidTimeseriesModelSIR import CovidTimeseriesModelSIR
import simplejson
import pandas as pd

from libs import build_params
from libs.datasets import JHUDataset
from libs.datasets import FIPSPopulation
from libs.datasets import DHBeds
from libs.datasets.dataset_utils import AggregationLevel

_logger = logging.getLogger(__name__)


def get_pool(num_cores=None) -> multiprocessing.Pool:
    if not num_cores:
        num_cores = max(multiprocessing.cpu_count() - 2, 1)

    return multiprocessing.Pool(num_cores)


def get_backfill_historical_estimates(df):

    CONFIRMED_HOSPITALIZED_RATIO = 4
    RECOVERY_SHIFT = 13
    HOSPITALIZATION_RATIO = 0.073

    df['estimated_recovered'] = df.cases.shift(RECOVERY_SHIFT).fillna(0)
    df['active'] = df.cases - (df.deaths + df.estimated_recovered)
    df['estimated_hospitalized'] = df['active']/CONFIRMED_HOSPITALIZED_RATIO
    df['estimated_infected'] = df['estimated_hospitalized']/HOSPITALIZATION_RATIO
    return df


def prepare_data_for_website(
    data, historicals, population, min_begin_date, max_end_date, interval: int = 4
):
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

    data["all_hospitalized"] = data["infected_b"] + data["infected_c"]
    data["all_infected"] = data["infected_a"] + data["infected_b"] + data["infected_c"]

    cols = [
        "date",
        "a", # total
        "b", # susceptible
        "c", # exposed
        "d", # infected
        "e", # infected_a (not hospitalized, but infected)
        "f", # infected_b (hospitalized not in icu)
        "g", # infected_c (in icu)
        "all_hospitalized", # infected_b + infected_c
        "all_infected", # infected_a + infected_b + infected_c
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
    website_ordering = website_ordering[
        website_ordering.index % interval == 0
    ].reset_index()

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
            "all_hospitalized": int,
            "all_infected": int,
            "dead": int,
            "beds": int,
            "population": int,
        }
    )
    website_ordering = website_ordering.astype(
        {
            "all_hospitalized": str,
            "all_infected": str,
            "dead": str,
            "beds": str,
            "population": str,
        }
    )
    historicals_df = get_backfill_historical_estimates(historicals)

    relevant_date_index = pd.to_datetime(website_ordering.date).isin(historicals_df.date)
    extract_real_date_index = historicals_df.date.isin(pd.to_datetime(website_ordering.date))

    website_ordering.loc[relevant_date_index, 'all_infected'] = (
        historicals_df[extract_real_date_index]['estimated_infected'].values
    )
    website_ordering.loc[relevant_date_index, 'all_hospitalized'] = (
        historicals_df[extract_real_date_index]['estimated_hospitalized'].values
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


def model_state(timeseries, starting_beds, population, interventions=None):

    # we should cut this, only used by the get_timeseries function, but probably not needed
    MODEL_INTERVAL = 4

    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    DATA_PARAMETERS = {
        "timeseries": timeseries,
        "beds": starting_beds,
        "population": population,
    }
    MODEL_PARAMETERS = {
        "model": "seir",
        "use_harvard_params": False,  # If True use the harvard parameters directly, if not calculate off the above
        "fix_r0": False,  # If True use the parameters that make R0 2.4, if not calculate off the above
        "days_to_model": 270,
        ## Variables for calculating model parameters Hill -> our names/calcs
        # IncubPeriod: Average incubation period, days - presymptomatic_period
        # DurMildInf: Average duration of mild infections, days - duration_mild_infections
        # FracMild: Average fraction of (symptomatic) infections that are mild - (1 - hospitalization_rate)
        # FracSevere: Average fraction of (symptomatic) infections that are severe - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # FracCritical: Average fraction of (symptomatic) infections that are critical - hospitalization_rate * hospitalized_cases_requiring_icu_care
        # CFR: Case fatality rate (fraction of infections that eventually result in death) - case_fatality_rate
        # DurHosp: Average duration of hospitalization (time to recovery) for individuals with severe infection, days - hospital_time_recovery
        # TimeICUDeath: Average duration of ICU admission (until death or recovery), days - icu_time_death
        # LOGIC ON INITIAL CONDITIONS:
        # hospitalized = case load from timeseries on last day of data / 4
        # mild = hospitalized / hospitalization_rate
        # icu = hospitalized * hospitalized_cases_requiring_icu_care
        # expoosed = exposed_infected_ratio * mild
        "presymptomatic_period": 3,  # Time before exposed are infectious, In days
        "duration_mild_infections": 6,  # Time mildly infected people stay sick before hospitalization or recovery, In days
        "hospital_time_recovery": 6,  # Duration of hospitalization before icu or recovery, In days
        "icu_time_death": 8,  # Time from ICU admission to death, In days
        "beta": 0.6,
        "beta_hospitalized": 0.1,
        "beta_icu": 0.1,
        "hospitalization_rate": 0.0727,
        "hospitalized_cases_requiring_icu_care": 0.1397,
        "case_fatality_rate": 0.0109341104294479,
        "exposed_from_infected": True,  # calculate the initial exposed based on infected?
        "exposed_infected_ratio": 1.2,
        "hospital_capacity_change_daily_rate": 1.05,
        "max_hospital_capacity_factor": 2.07,
        "initial_hospital_bed_utilization": 0.6,
        "interventions": interventions,
        "observed_daily_growth_rate": 1.21,
    }

    MODEL_PARAMETERS["beta"] = (
        0.3 + ((MODEL_PARAMETERS["observed_daily_growth_rate"] - 1.09) / 0.02) * 0.05
    )

    MODEL_PARAMETERS["case_fatality_rate_hospitals_overwhelmed"] = (
        MODEL_PARAMETERS["hospitalization_rate"]
        * MODEL_PARAMETERS["hospitalized_cases_requiring_icu_care"]
    )

    MODEL_PARAMETERS.update(DATA_PARAMETERS)

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


def build_county_summary(
    min_date: datetime.datetime,
    output_dir: pathlib.Path,
    country: str = "USA",
    state: str = None,
):
    """Builds county summary json files."""
    beds_data = DHBeds.local().beds()
    population_data = FIPSPopulation.local().population()
    timeseries = JHUDataset.local().timeseries()
    timeseries = timeseries.get_subset(
        AggregationLevel.COUNTY, after=min_date, country=country, state=state
    )

    _logger.info(f"Outputting to {output_dir}")
    if not output_dir.exists():
        _logger.info(f"{output_dir} does not exist, creating")
        output_dir.mkdir(parents=True)

    counties_by_state = defaultdict(list)
    for country, state, county, fips in timeseries.county_keys():
        counties_by_state[state].append((county, fips))

    all_data = {"counties_with_data": []}
    for state, counties in counties_by_state.items():
        data = {"counties_with_data": []}
        for county, fips in counties:
            cases = timeseries.get_data(state=state, country=country, fips=fips)
            beds = beds_data.get_county_level(state, fips=fips)
            population = population_data.get_county_level(country, state, fips=fips)
            if population and beds and sum(cases.cases):
                data["counties_with_data"].append(fips)
                all_data["counties_with_data"].append(fips)

        output_path = output_dir / f"{state}.summary.json"
        output_path.write_text(json.dumps(data, indent=2))

    output_path = output_dir / "fips_summary.json"
    output_path.write_text(json.dumps(all_data, indent=2))


def catch_and_return_errors(func) -> Tuple[Optional[Exception], Optional[Any]]:
    """Catch and report errors, wrapping response in a tuple with error and result.

    Args:
        func: Function to wrap.

    Returns: Tuple of optional error and optional result.
    """
    @functools.wraps(func)
    def run(*args, **kwargs):

        try:
            result = func(*args, **kwargs)
            return (None, result)
        except Exception as err:
            error_msg = traceback.format_exc()
            return ((err, error_msg), None)

    return run


def _result_callback_wrapper(key):
    """Creates a callback function to log errors on parallel task completion.

    Args:
        key: Key used to help identify inputs that caused error.

    Returns: Callback function.
    """
    def callback(result):
        error, _ = result

        if error:
            exc, error_traceback = error
            _logger.error(f"Failed to process: {key}")
            _logger.error(error_traceback)

    return callback


@catch_and_return_errors
def forecast_each_state(
    country,
    state,
    timeseries,
    beds_data,
    population_data,
    min_date,
    max_date,
    output_dir,
):
    _logger.info(f"Generating data for state: {state}")
    cases = timeseries.get_data(state=state)
    beds = beds_data.get_state_level(state)
    if not beds:
        _logger.error(f"Failed to get beds data for {state}")
        return
    population = population_data.get_state_level(country, state)
    if not population:
        _logger.warning(f"Missing population for {state}")
        return

    for i, intervention in enumerate(build_params.get_interventions()):
        _logger.info(f"Running intervention {i} for {state}")
        results = model_state(cases, beds, population, intervention)
        website_data = prepare_data_for_website(
            results, cases, population, min_date, max_date, interval=4
        )

        write_results(website_data, output_dir, f"{state}.{i}.json")


@catch_and_return_errors
def forecast_each_county(
    min_date,
    max_date,
    country,
    state,
    county,
    fips,
    timeseries,
    beds_data,
    population_data,
    output_dir,
):
    _logger.info(f"Running model for county: {county}, {state} - {fips}")
    cases = timeseries.get_data(state=state, country=country, fips=fips)
    beds = beds_data.get_county_level(state, fips=fips)
    population = population_data.get_county_level(country, state, fips=fips)
    total_cases = sum(cases.cases)
    if not population or not beds or not total_cases:
        _logger.debug(
            f"Missing data, skipping: Beds: {beds} Pop: {population} Total Cases: {total_cases}"
        )
        return

    _logger.info(
        f"Running interventions for {county}, {state}: {fips} - "
        f"total cases: {total_cases} beds: {beds} pop: {population}"
    )

    interventions = build_params.get_interventions()
    for i, intervention in enumerate(interventions):
        results = model_state(cases, beds, population, intervention)
        website_data = prepare_data_for_website(
            results, cases, population, min_date, max_date, interval=4
        )
        write_results(website_data, output_dir, f"{state}.{fips}.{i}.json")


def run_county_level_forecast(
    min_date: datetime.datetime,
    max_date: datetime.datetime,
    output_dir: pathlib.Path,
    country: str = "USA",
    state: str = None,
):
    beds_data = DHBeds.local().beds()
    population_data = FIPSPopulation.local().population()
    timeseries = JHUDataset.local().timeseries()
    timeseries = timeseries.get_subset(
        AggregationLevel.COUNTY, after=min_date, country=country, state=state
    )

    _logger.info(f"Outputting to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    counties_by_state = defaultdict(list)
    county_keys = timeseries.county_keys()
    for country, state, county, fips in county_keys:
        counties_by_state[state].append((county, fips))

    pool = get_pool()
    for state, counties in counties_by_state.items():
        _logger.info(f"Running county models for {state}")
        for county, fips in counties:
            args = (
                min_date,
                max_date,
                country,
                state,
                county,
                fips,
                timeseries,
                beds_data,
                population_data,
                output_dir,
            )

            pool.apply_async(
                forecast_each_county,
                args,
                callback=_result_callback_wrapper(f"{county}, {state}: {fips}")
            )

    pool.close()
    pool.join()


def run_state_level_forecast(
    min_date, max_date, output_dir, country="USA", state=None,
):
    # DH Beds dataset does not have all counties, so using the legacy state
    # level bed data.
    beds_data = DHBeds.local().beds()
    population_data = FIPSPopulation.local().population()
    timeseries = JHUDataset.local().timeseries()
    timeseries = timeseries.get_subset(
        AggregationLevel.STATE, after=min_date, country=country, state=state
    )
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = get_pool()
    for state in timeseries.states:
        args = (
            country,
            state,
            timeseries,
            beds_data,
            population_data,
            min_date,
            max_date,
            output_dir,
        )
        pool.apply_async(
            forecast_each_state,
            args,
            callback=_result_callback_wrapper(f"{state}, {country}")
        )

    pool.close()
    pool.join()


if __name__ == "__main__":
    _logger.warning(
        "Models are no longer ran from run.py. Please see README.md for the most up to date "
        "way to run models."
    )
