import enum

# Fips code chosen for all unknown fips values.
# TODO: This should maybe be unique per state.
UNKNOWN_FIPS = "99999"


class Intervention(enum.Enum):
    NO_INTERVENTION = 0
    FLATTEN = 1
    # FULL_CONTAINMENT = 2 # you are cancelled
    SOCIAL_DISTANCING = 3
    CURRENT = 4  # look at what the state is and get the file for that
    INFERRED = 5 # given the previous pattern, how do we predict going forward

    @classmethod
    def from_webui_data_adaptor(cls, label):
        if label == "suppression_policy__no_intervention": 
            return cls.NO_INTERVENTION
        elif label == "suppression_policy__flatten_the_curve":
            return cls.FLATTEN
        elif label == "suppression_policy__inferred": 
            return cls.INFERRED
        elif label == "suppression_policy__social_distancing":
            return cls.SOCIAL_DISTANCING
        raise Exception(f"Unexpected WebUI Data Adaptor label: {label}")

    @classmethod
    def from_str(cls, label):
        if label == "shelter_in_place":
            return cls.FLATTEN
        elif label == "social_distancing":
            return cls.SOCIAL_DISTANCING
        else:
            return cls.NO_INTERVENTION
