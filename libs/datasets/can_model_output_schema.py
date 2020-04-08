"""
Ouput of the 
"""
CAN_MODEL_OUTPUT_SCHEMA = [
    "day_num",
    # ^ for index column
    "date",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "all_hospitalized",
    "all_infected",
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

# Exclude the unnamed columns from our data frames until we have a use for them
CAN_MODEL_OUTPUT_SCHEMA_EXCLUDED_COLUMNS = ["a","b","c","d","e","f","g","i","j","k","l","m","n"]
