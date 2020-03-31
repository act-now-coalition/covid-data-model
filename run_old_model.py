import logging
import pandas as pd
import datetime
import time
import simplejson
from libs.build_params import r0, OUTPUT_DIR, INTERVENTIONS
from libs.CovidTimeseriesModel import CovidTimeseriesModel
from libs.CovidDatasets import CDSDataset, JHUDataset

_logger = logging.getLogger(__name__)

def record_results(
    res, directory, name, num, pop, min_begin_date=None, max_end_date=None
):
    import os.path

    # Indexes used by website JSON:
    # date: 0,
    # hospitalizations: 8,
    # cumulativeInfected: 9,
    # cumulativeDeaths: 10,
    # beds: 11,
    # totalPopulation: 16,

    cols = [
        "Date",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "Pred. Hosp.",
        "Cum. Inf.",
        "Cum. Deaths",
        "Avail. Hosp. Beds",
        "i",
        "j",
        "k",
        "l",
        "population",
        "m",
        "n",
    ]

    website_ordering = pd.DataFrame(res, columns=cols).fillna(0)

    if min_begin_date is not None:
        website_ordering = pd.DataFrame(
            website_ordering[website_ordering["Date"] >= min_begin_date]
        )
    if max_end_date is not None:
        website_ordering = pd.DataFrame(
            website_ordering[website_ordering["Date"] <= max_end_date]
        )

    website_ordering["Date"] = website_ordering["Date"].dt.strftime("%-m/%-d/%y")
    website_ordering["population"] = pop
    website_ordering = website_ordering.astype(
        {
            "Pred. Hosp.": int,
            "Cum. Inf.": int,
            "Cum. Deaths": int,
            "Avail. Hosp. Beds": int,
            "population": int,
        }
    )
    website_ordering = website_ordering.astype(
        {
            "Pred. Hosp.": str,
            "Cum. Inf.": str,
            "Cum. Deaths": str,
            "Avail. Hosp. Beds": str,
            "population": str,
        }
    )

    with open(
        os.path.join(directory, name.upper() + "." + str(num) + ".json").format(name),
        "w",
    ) as out:
        simplejson.dump(website_ordering.values.tolist(), out, ignore_nan=True)

    # @TODO: Remove once the frontend no longer expects some states to be lowercase.
    with open(
        os.path.join(directory, name.lower() + "." + str(num) + ".json").format(name),
        "w",
    ) as out:
        simplejson.dump(website_ordering.values.tolist(), out, ignore_nan=True)

def model_state(dataset, country, state, starting_beds, interventions=None):
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
        'beds': starting_beds,
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
        'estimated_new_cases_per_death': 180,
        'estimated_new_cases_per_confirmed': 4
    }
    return CovidTimeseriesModel().forecast(model_parameters=MODEL_PARAMETERS)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # @TODO: Remove interventions override once support is in the Harvard model.
    country = "USA"
    min_date = datetime.datetime(2020, 3, 7)
    max_date = datetime.datetime(2020, 7, 6)

    dataset = JHUDataset()
    for state in dataset.get_all_states_by_country(country):
        logging.info("Generating data for state: {}".format(state))
        starting_beds = dataset.get_beds_by_country_state(country, state)
        for i in range(0, len(INTERVENTIONS)):
            _logger.info(f"Running intervention {i} for {state}")
            intervention = INTERVENTIONS[i]
            results = model_state(dataset, country, state, starting_beds, intervention)
            record_results(
                results,
                OUTPUT_DIR,
                state,
                i,
                dataset.get_population_by_country_state(country, state),
                min_date,
                max_date,
            )
