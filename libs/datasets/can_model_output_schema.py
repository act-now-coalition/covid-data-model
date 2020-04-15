"""
Ouput of the 
"""
DATE = "date"
TOTAL = "total"
TOTAL_SUSCEPTIBLE = "susceptible"
EXPOSED = "exposed"
INFECTED = "infected"
INFECTED_A = "infected_a"
INFECTED_B = "infected_b"
INFECTED_C = "infected_c"
ALL_HOSPITALIZED = "all_hospitalized"
ALL_INFECTED = "all_infected"
DEAD = "dead"
BEDS = "beds"
POPULATION = "population"

CAN_MODEL_OUTPUT_SCHEMA = [
    "day_num",
    # ^ for index column
    DATE,
    TOTAL,
    TOTAL_SUSCEPTIBLE,
    EXPOSED,
    INFECTED, 
    INFECTED_A, # (not hospitalized, but infected)
    INFECTED_B, # infected_b (hospitalized not in icu)
    INFECTED_C, # infected_c (in icu)
    ALL_HOSPITALIZED, # infected_b + infected_c
    ALL_INFECTED, # infected_a + infected_b + infected_c
    DEAD,
    BEDS,
    "i",
    "j",
    "k",
    "l",
    POPULATION,
    "m",
    "n",
]

# Exclude the unnamed columns from our data frames until we have a use for them
CAN_MODEL_OUTPUT_SCHEMA_EXCLUDED_COLUMNS = ["i","j","k","l","m","n"]
