import enum

# Fips code chosen for all unknown fips values.
# TODO: This should maybe be unique per state.
UNKNOWN_FIPS = "99999"


class Intervention(enum.Enum):
    NO_INTERVENTION = 0
    FLATTEN = 1
    FULL_CONTAINMENT = 2
    SOCIAL_DISTANCING = 3
    CURRENT = 4  # look at what the state is and get the file for that

    @classmethod
    def from_str(cls, label):
        if label == "shelter_in_place":
            return cls.FLATTEN
        elif label == "social_distancing":
            return cls.SOCIAL_DISTANCING
        elif label == "lockdown":
            return cls.FULL_CONTAINMENT
        else:
            return cls.NO_INTERVENTION
