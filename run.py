from libs.CovidDatasets import CDSDataset, JHUDataset
from libs.CovidTimeseriesModelSIR import CovidTimeseriesModelSIR
from libs.build_params import r0, OUTPUT_DIR, INTERVENTIONS
import simplejson
import time
import datetime
import pandas as pd
import logging

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

    website_ordering = pd.DataFrame(res, columns=cols).fillna(0)

    # @TODO: Find a better way of restricting to every fourth day.
    #        Alternatively, change the website's expectations.
    website_ordering = pd.DataFrame(website_ordering[website_ordering.index % 4 == 0])

    if min_begin_date is not None:
        website_ordering = pd.DataFrame(
            website_ordering[website_ordering["date"] >= min_begin_date]
        )
    if max_end_date is not None:
        website_ordering = pd.DataFrame(
            website_ordering[website_ordering["date"] <= max_end_date]
        )

    website_ordering["date"] = website_ordering["date"].dt.strftime("%-m/%-d/%y")
    website_ordering["population"] = pop
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

    # Constants
    start_time = time.time()
    HOSPITALIZATION_RATE = 0.0727
    HOSPITALIZED_CASES_REQUIRING_ICU_CARE = 0.1397
    TOTAL_INFECTED_PERIOD = 12
    MODEL_INTERVAL = 4
    POP = dataset.get_population_by_country_state(country, state)
    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    MODEL_PARAMETERS = {
        # Pack the changeable model parameters
        "timeseries": dataset.get_timeseries_by_country_state(
            country, state, MODEL_INTERVAL
        ),
        "beds": dataset.get_beds_by_country_state(country, state),
        "population": POP,
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
