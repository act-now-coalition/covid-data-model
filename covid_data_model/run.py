from typing import List
import os
import copy
import datetime
from collections import namedtuple
from collections import defaultdict
import logging
import simplejson
from covid_data_model import covid_timeseries_model
from covid_data_model import covid_datasets

_logger = logging.getLogger(__name__)

Intervention = namedtuple("Intervention", ["name", "r0_by_date"])


def record_results(res, directory, name, num, pop):
    vals = copy.copy(res)
    # Format the date in the manner the front-end expects
    vals["Date"] = res["Date"].apply(lambda d: f"{d.month}/{d.day}/{d.year}")
    # Set the population
    vals["Population"] = pop
    # Write the results to the specified directory
    output_path = os.path.join(directory, name.upper() + "." + str(num) + ".json")
    output_columns = [
        "Date",
        "R",
        "Beg. Susceptible",
        "New Inf.",
        "Curr. Inf.",
        "Recov. or Died",
        "End Susceptible",
        "Actual Reported",
        "Pred. Hosp.",
        "Cum. Inf.",
        "Cum. Deaths",
        "Avail. Hosp. Beds",
        "S&P 500",
        "Est. Actual Chance of Inf.",
        "Pred. Chance of Inf.",
        "Cum. Pred. Chance of Inf.",
        "Population",
        "R0",
        "% Susceptible",
    ]
    with open(output_path, "w") as out:
        simplejson.dump(vals[output_columns].values.tolist(), out, ignore_nan=True)


def model_state(country, state, interventions=None):
    hospitalization_rate = 0.0727
    hospitalized_cases_requiring_icu_care = 0.1397
    total_infected_period = 12
    model_interval = 4
    r0 = 2.4
    dataset = covid_datasets.CovidDatasets(filter_past_date=datetime.date(2020, 3, 19))
    pop = dataset.get_population_by_country_state(country, state)
    # Pack all of the assumptions and parameters into a dict that can be passed into the model
    model_parameters = {
        # Pack the changeable model parameters
        "timeseries": dataset.get_timeseries_by_country_state(
            country, state, model_interval
        ),
        "beds": dataset.get_beds_by_country_state(country, state),
        "population": pop,
        "projection_iterations": 24,  # Number of iterations into the future to project
        "r0": r0,
        "interventions": interventions,
        "hospitalization_rate": hospitalization_rate,
        "initial_hospitalization_rate": 0.05,
        "case_fatality_rate": 0.0109341104294479,
        "hospitalized_cases_requiring_icu_care": hospitalized_cases_requiring_icu_care,
        # Assumes that anyone who needs ICU care and doesn't get it dies
        "case_fatality_rate_hospitals_overwhelmed": hospitalization_rate
        * hospitalized_cases_requiring_icu_care,
        "hospital_capacity_change_daily_rate": 1.05,
        "max_hospital_capacity_factor": 2.07,
        "initial_hospital_bed_utilization": 0.6,
        "model_interval": 4,  # In days
        "total_infected_period": 12,  # In days
        "rolling_intervals_for_current_infected": int(
            round(total_infected_period / model_interval, 0)
        ),
    }

    timeseries_model = covid_timeseries_model.CovidTimeseriesModel(model_parameters)
    return timeseries_model.forecast_region()


def main(
    interventions: List[Intervention], output_folder: str, states: List[str] = None
):
    """Run intervention models.

    Args:
        interventions: List of Interventions to run.
        output_folder: Directory to write results to.
        states: Optional list of states to run model on.
    """
    dataset = covid_datasets.CovidDatasets()
    states = states or dataset.get_all_states_by_country("USA")

    results = defaultdict(dict)

    for state in states:
        for i, intervention in enumerate(interventions):
            _logger.info(f"Running {intervention.name} on {state}")
            state_model_results = model_state(
                "USA", state, interventions=intervention.r0_by_date
            )
            state_population = dataset.get_population_by_country_state("USA", state)
            record_results(
                state_model_results, output_folder, state, i, state_population
            )
            results[state][intervention.name] = state_model_results

    return results


r0 = 2.4

INTERVENTIONS = [
    Intervention("No Intervention", None),
    Intervention(
        "Flatten the Curve",
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 4, 20): 1.1,
            datetime.date(2020, 5, 22): 0.8,
            datetime.date(2020, 6, 23): r0,
        },
    ),
    Intervention(
        "Full Containment",
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 3, 31): 0.3,
            datetime.date(2020, 4, 28): 0.2,
            datetime.date(2020, 5, 6): 0.1,
            datetime.date(2020, 5, 10): 0.35,
            datetime.date(2020, 5, 18): r0,
        },
    ),
    Intervention(
        "@TODO: Model w/ FlatteningTheCurve (2 wk delay)",
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 4, 20): 1.1,
            datetime.date(2020, 5, 22): 0.8,
            datetime.date(2020, 6, 23): r0,
        },
    ),
    Intervention(
        "@TODO: Model w/ FlatteningTheCurve (1 mo delay)",
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 4, 20): 1.1,
            datetime.date(2020, 5, 22): 0.8,
            datetime.date(2020, 6, 23): r0,
        },
    ),
    Intervention(
        "@TODO: Full Containment (1 wk dly)",
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 3, 31): 0.3,
            datetime.date(2020, 4, 28): 0.2,
            datetime.date(2020, 5, 6): 0.1,
            datetime.date(2020, 5, 10): 0.35,
            datetime.date(2020, 5, 18): r0,
        },
    ),
    Intervention(
        "Full Containment (2 wk dly)",
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 3, 31): 0.3,
            datetime.date(2020, 4, 28): 0.2,
            datetime.date(2020, 5, 6): 0.1,
            datetime.date(2020, 5, 10): 0.35,
            datetime.date(2020, 5, 18): r0,
        },
    ),
    Intervention(
        "Social Distancing",
        {datetime.date(2020, 3, 23): 1.7, datetime.date(2020, 6, 23): r0},
    ),
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main(INTERVENTIONS, "results/test")
