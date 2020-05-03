
class CommonIndexFields(object):
    # Column for FIPS code. Right now a column containing fips data may be
    # county fips (a length 5 string) or state fips (a length 2 string).
    FIPS = "fips"

    # 2 letter state abbreviation, i.e. MA
    STATE = "state"

    COUNTRY= "country"

    AGGREGATE_LEVEL = "aggregate_level"

    DATE = "date"


class CommonFields(object):
    """Common field names shared across different sources of data"""

    FIPS = "fips"

    # 2 letter state abbreviation, i.e. MA
    STATE = "state"

    COUNTRY= "country"

    COUNTY= "county"

    AGGREGATE_LEVEL = "aggregate_level"

    DATE = "date"

    # Full state name, i.e. Massachusetts
    STATE_FULL_NAME = "state_full_name"

    CASES = "cases"
    DEATHS = "deaths"
    RECOVERED = "recovered"
    CUMULATIVE_HOSPITALIZED = "cumulative_hospitalized"
    CUMULATIVE_ICU = "cumulative_icu"

    POSITIVE_TESTS = "positive_tests"
    NEGATIVE_TESTS = "negative_tests"

    # Current values
    CURRENT_ICU = "current_icu"
    CURRENT_HOSPITALIZED = "current_hospitalized"
    CURRENT_VENTILATED = "current_ventilated"

    POPULATION = "population"

    STAFFED_BEDS = "staffed_beds"
    LICENSED_BEDS = "licensed_beds"
    ICU_BEDS = "icu_beds"
    ALL_BED_TYPICAL_OCCUPANCY_RATE = "all_beds_occupancy_rate"
    ICU_TYPICAL_OCCUPANCY_RATE = "icu_occupancy_rate"
    MAX_BED_COUNT = "max_bed_count"
