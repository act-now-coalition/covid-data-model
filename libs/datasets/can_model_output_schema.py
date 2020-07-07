"""
One data model output of the modeling pipeline to serve to the API.
"""
FIPS = "fips"
INTERVENTION = "intervention"
DAY_NUM = "day_num"  # Index Column. Generally not the same between simulations.
DATE = "date"  # Date in the timeseries.
TOTAL = "total"  # All people in the model. This should always be population.
TOTAL_SUSCEPTIBLE = "susceptible"
EXPOSED = "exposed"
INFECTED = "infected"
INFECTED_A = "infected_a"
INFECTED_B = "infected_b"
INFECTED_C = "infected_c"
ALL_HOSPITALIZED = "all_hospitalized"
ALL_INFECTED = "all_infected"
DEAD = "dead"
BEDS = "beds"  # General bed capacity excluding ICU beds.
CUMULATIVE_INFECTED = "cumulative_infected"
Rt = "Rt"  # Effective reproduction number at time t.
Rt_ci90 = "Rt_ci90"  # 90% confidence interval at time t.
CURRENT_VENTILATED = "current_ventilated"
POPULATION = "population"
ICU_BED_CAPACITY = "icu_bed_capacity"
VENTILATOR_CAPACITY = "ventilator_capacity"
RT_INDICATOR = "Rt_indicator"
RT_INDICATOR_CI90 = "Rt_indicator_ci90"

CAN_MODEL_OUTPUT_SCHEMA = [
    DAY_NUM,
    # ^ for index column
    DATE,
    TOTAL,
    TOTAL_SUSCEPTIBLE,
    EXPOSED,
    INFECTED,
    INFECTED_A,  # (not hospitalized, but infected)
    INFECTED_B,  # infected_b (hospitalized not in icu)
    INFECTED_C,  # infected_c (in icu)
    ALL_HOSPITALIZED,  # infected_b + infected_c
    ALL_INFECTED,  # infected_a + infected_b + infected_c
    DEAD,
    BEDS,
    CUMULATIVE_INFECTED,
    Rt,
    Rt_ci90,
    CURRENT_VENTILATED,
    POPULATION,
    ICU_BED_CAPACITY,
    VENTILATOR_CAPACITY,
    RT_INDICATOR,
    RT_INDICATOR_CI90,
]

# Exclude the unnamed columns from our data frames until we have a use for them
CAN_MODEL_OUTPUT_SCHEMA_EXCLUDED_COLUMNS = []
