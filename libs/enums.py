import enum

# Fips code chosen for all unknown fips values.
# TODO: This should maybe be unique per state.
UNKNOWN_FIPS = "99999"

class Intervention(enum.Enum):
    NO_INTERVENTION = 0
    FLATTEN = 1 # on the webiste, strictDistancingNow
    # FULL_CONTAINMENT = 2 # you are cancelled, but reusing this enum value
    SOCIAL_DISTANCING = 3 # weak distancingNow on the website
    CURRENT = 4  # look at what the state is and get the file for that
    # We are using enum 2 for consistency with the website 
    INFERRED = 2 # given the previous pattern, how do we predict going forward

    @classmethod
    def county_supported_interventions(cls): 
        return [
            Intervention.NO_INTERVENTION,
            Intervention.FLATTEN,
            Intervention.SOCIAL_DISTANCING,
            Intervention.CURRENT
        ]

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
