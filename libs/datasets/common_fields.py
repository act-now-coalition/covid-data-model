

class CommonFields(object):
    """Common field names shared across different sources of data"""

    # Column for FIPS code. Right now a column containing fips data may be
    # county fips (a length 5 string) or state fips (a length 2 string).
    FIPS = "fips"

    # 2 letter state abbreviation, i.e. MA
    STATE = "state"

    # Full state name, i.e. Massachusetts
    STATE_FULL_NAME = "state_full_name"
