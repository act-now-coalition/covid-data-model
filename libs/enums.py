import enum

# Fips code chosen for all unknown fips values.
# TODO: This should maybe be unique per state.
UNKNOWN_FIPS = "99999"

class Intervention(enum.Enum):
    NO_INTERVENTION = 0
    FLATTEN = 1
    FULL_CONTAINMENT = 2
    SOCIAL_DISTANCING = 3