import us
import structlog
from datetime import timedelta

from pyseir import load_data

log = structlog.get_logger()

UNITY_SCALING_FACTOR = 1


def _get_model_to_dataset_conversion_factors_for_state(state, t0_simulation, fips):
    """
    Return scaling factors to convert model hospitalization and model icu numbers to match
    the most current values provided in combined_datasets.

    Parameters
    ----------
    state
    t0_simulation
    fips

    Returns
    -------
    convert_model_to_observed_hospitalized
    convert_model_to_observed_icu
    """

    # Get "Ground Truth" from outside datasets
    # NB: If only cumulatives are provided, we estimate current load. So this isn't strictly
    # actuals from covid-tracking.
    state_abbreviation = us.states.lookup(state).abbr
    days_since_start, observed_latest_hospitalized = load_data.get_current_hospitalized_for_state(
        state=state_abbreviation,
        t0=t0_simulation,
        category=load_data.HospitalizationCategory.HOSPITALIZED,
    )

    if observed_latest_hospitalized is None:
        # We have no observed data available. Best we can do is pass unity factors.
        return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
    elif observed_latest_hospitalized == 0:
        # Right now our scaling factor can not capture this edge case
        log.msg(
            "Observed Hospitalized was 0 so we can not scale model outputs to latest observed",
            state=state,
        )
        return UNITY_SCALING_FACTOR, UNITY_SCALING_FACTOR
    else:
        # Let's try to get a conversion for model to observed hospitalization

        # Rebuild date object
        t_latest_hosp_data_date = t0_simulation + timedelta(days=int(days_since_start))

        # Get Compartment Values for a Given Time
        model_state_hosp_gen = load_data.get_compartment_value_on_date(
            fips=fips, compartment="HGen", date=t_latest_hosp_data_date
        )
        model_state_hosp_icu = load_data.get_compartment_value_on_date(
            fips=fips, compartment="HICU", date=t_latest_hosp_data_date
        )

        # In the model, general hospital and icu hospital are disjoint states. We have to add them
        # together to get the correct comparable for hospitalized.
        model_heads_in_beds = model_state_hosp_gen + model_state_hosp_icu

        model_to_observed_hospitalized_ratio = observed_latest_hospitalized / model_heads_in_beds

        # Now let's look at ICU observed data
        _, observed_latest_icu = load_data.get_current_hospitalized_for_state(
            state=state_abbreviation,
            t0=t0_simulation,
            category=load_data.HospitalizationCategory.ICU,
        )
        if observed_latest_icu is None:
            # We have observed hospitalizations, but not observed icu
            # We therefore scale ICU the same as general hospitalization
            model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
        elif observed_latest_icu == 0:
            # Right now our scaling factor can not capture this edge case
            log.msg(
                "Observed ICU was 0. Falling back on Observed Hospitalization Ratio", state=state
            )
            model_to_observed_icu_ratio = model_to_observed_hospitalized_ratio
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio
        else:
            # We will have separate scaling factors. This is predicated on the assumption that we
            # should impose the location specific relative scaling factors instead of the model
            # derived ratio.
            model_to_observed_icu_ratio = observed_latest_icu / model_state_hosp_icu
            return model_to_observed_hospitalized_ratio, model_to_observed_icu_ratio


def _get_model_to_dataset_conversion_factors_for_county(state, t0_simulation, fips):
    """"""
    return NotImplementedError


if __name__ == "__main__":
    # Run those unittests
    pass
