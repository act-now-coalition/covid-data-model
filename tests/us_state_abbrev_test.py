from libs.us_state_abbrev import (
    US_STATE_ABBREV,
    ABBREV_US_STATE,
    ABBREV_US_FIPS,
    ABBREV_US_UNKNOWN_COUNTY_FIPS,
)


def test_state_name_fips_abbrev_maps():
    assert US_STATE_ABBREV["Wisconsin"] == "WI"
    assert ABBREV_US_STATE["WI"] == "Wisconsin"
    assert ABBREV_US_FIPS["TX"] == "48"

    # Number of entries (50 states, DC, 5 Territories) == 56?
    assert 56 == len(US_STATE_ABBREV)

    assert ABBREV_US_UNKNOWN_COUNTY_FIPS["CA"] == "06999"
