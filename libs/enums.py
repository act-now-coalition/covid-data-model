import enum

# Fips code chosen for all unknown fips values.
# TODO: This should maybe be unique per state.
UNKNOWN_FIPS = "99999"

class Intervention(enum.Enum):
    NO_INTERVENTION = 0
    STRONG_INTERVENTION = 1 # on the webiste, strictDistancingNow
    WEAK_INTERVENTION = 3 # weak distancingNow on the website
    CURRENT_INTERVENTION = 4  # look at what the state is and get the file for that
    # We are using enum 2 for consistency with the website
    OBSERVED_INTERVENTION = 2 # given the previous pattern, how do we predict going forward

    @classmethod
    def county_supported_interventions(cls):
        return [
            Intervention.NO_INTERVENTION,
            Intervention.STRONG_INTERVENTION,
            Intervention.WEAK_INTERVENTION,
            Intervention.CURRENT_INTERVENTION,
        ]

    @classmethod
    def from_webui_data_adaptor(cls, label):
        if label == "suppression_policy__no_intervention":
            return cls.NO_INTERVENTION
        elif label == "suppression_policy__flatten_the_curve":
            return cls.STRONG_INTERVENTION
        elif label == "suppression_policy__inferred":
            return cls.OBSERVED_INTERVENTION
        elif label == "suppression_policy__social_distancing":
            return cls.WEAK_INTERVENTION
        raise Exception(f"Unexpected WebUI Data Adaptor label: {label}")

    @classmethod
    def from_str(cls, label):
        if label == "shelter_in_place":
            return cls.STRONG_INTERVENTION
        elif label == "social_distancing":
            return cls.WEAK_INTERVENTION
        else:
            return cls.NO_INTERVENTION
