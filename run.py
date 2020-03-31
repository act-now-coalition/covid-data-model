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

    # we should cut this, only used by the get_timeseries function, but probably not needed
    MODEL_INTERVAL = 4

    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    DATA_PARAMETERS = {
        "timeseries": dataset.get_timeseries_by_country_state(
            country, state, MODEL_INTERVAL
        ),
        "beds": dataset.get_beds_by_country_state(country, state),
        "population": dataset.get_population_by_country_state(country, state),
    }

    MODEL_PARAMETERS = {
        "model": "seir",
        "use_harvard_params": False,  # If True use the harvard parameters directly, if not calculate off the above
        "fix_r0": False,  # If True use the parameters that make R0 2.4, if not calculate off the above
        "days_to_model": 270,
        "hospitalization_rate": 0.0727,
        #'hospitalization_rate': 0.2,
        "hospitalized_cases_requiring_icu_care": 0.1397,
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
        "duration_mild_infections": 6,  # Time mildly infected poeple stay sick, In days
        "hospital_time_recovery": 11,  # Duration of hospitalization, In days
        "icu_time_death": 7,  # Time from ICU admission to death, In days
        "beta": 0.5,
        "beta_hospitalized": 0.1,
        "beta_icu": 0.1,
        "case_fatality_rate": 0.0109341104294479,
        "exposed_from_infected": True,
        "exposed_infected_ratio": 1,
        "hospital_capacity_change_daily_rate": 1.05,
        "max_hospital_capacity_factor": 2.07,
        "initial_hospital_bed_utilization": 0.6,
        "interventions": interventions,
    }

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
