import enum

# Fips code chosen for all unknown fips values.
# TODO: This should maybe be unique per state.
UNKNOWN_FIPS = "99999"

class Intervention(enum.Enum):
    NO_MITIGATION = 0
    HIGH_MITIGATION = 1 # on the webiste, strictDistancingNow
    MODERATE_MITIGATION = 3 # weak distancingNow on the website
    SELECTED_MITIGATION = 4  # look at what the state is and get the file for that
    # We are using enum 2 for consistency with the website
    OBSERVED_MITIGATION = 2 # given the previous pattern, how do we predict going forward

    @classmethod
    def county_supported_interventions(cls):
        return [
            Intervention.NO_MITIGATION,
            Intervention.HIGH_MITIGATION,
            Intervention.MODERATE_MITIGATION,
            Intervention.SELECTED_MITIGATION,
        ]

    @classmethod
    def from_webui_data_adaptor(cls, label):
        if label == "suppression_policy__no_intervention":
            return cls.NO_MITIGATION
        elif label == "suppression_policy__flatten_the_curve":
            return cls.HIGH_MITIGATION
        elif label == "suppression_policy__inferred":
            return cls.OBSERVED_MITIGATION
        elif label == "suppression_policy__social_distancing":
            return cls.MODERATE_MITIGATION
        raise Exception(f"Unexpected WebUI Data Adaptor label: {label}")

    @classmethod
    def from_str(cls, label):
        if label == "shelter_in_place":
            return cls.HIGH_MITIGATION
        elif label == "social_distancing":
            return cls.MODERATE_MITIGATION
        else:
            return cls.NO_MITIGATION
