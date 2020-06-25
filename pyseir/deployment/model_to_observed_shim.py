import numpy as np


def calculate_strict_shim(model: float, observed: float, log) -> float:
    """
    Take model outputs and calculate a "shim" value that can be added to the entire modeled
    series to make it match the latest (i.e. today's) actual reported value.
    As a consequence, this will shift any future forecasted values by the same amount.

    Parameters
    ----------
    model
        model estimate for latest date
    observed
        observed value for latest date
    log
        Log Instance
    Return
    ------
    shim: float
        Value to shim the timeseries by
    """

    # There are inconsistent None/"NaN" -> force all to np.nan for this scope.
    if observed is None:
        observed = np.nan

    if np.isnan(observed):
        shim = 0
    elif observed == 0:
        # As of 19 June 2020, the observed dataset is still being validated for erroneous inputs.
        # This includes cases of returning a 0 when we should be returning a np.nan/None.
        # For now, we will not apply a shim if the result returned is 0.
        shim = 0
    else:
        shim = observed - model

    log.info(
        event="calculate_strict_shim", shim=np.round(shim), observed=observed, model=np.round(model)
    )
    return shim


def calculate_intralevel_icu_shim(
    model_acute: float, model_icu: float, observed_icu: float, observed_total_hosps: float, log,
) -> float:
    """
    Take model outputs and calculate a "shim" value that can be added to the icu series to make it
    match the latest (i.e. today's) actual icu.

    Parameters
    ----------
    model_acute: float
        model estimate value for acute hospitalized
    model_icu: float
        model estimate value for icu hospitalized
    observed_icu: float
        observed value for icu hospitalized
    observed_total_hosps: float
        observed value for total hospitalized
    log
        Log Instance
    Return
    ------
    shim: float
        Value to shim icu data by
    """
    # There are inconsistent None/"NaN" -> force all to np.nan for this scope.
    if observed_icu is None:
        observed_icu = np.nan

    if np.isnan(observed_icu):
        # We don't have ICU specific data but let's try to have a shim informed by the shim used
        # for total hospitalization. This will maintain the same icu/total_hosp ratio that was
        # originally in the model.
        model_total_hosps_latest = model_acute + model_icu
        total_hosp_shim = calculate_strict_shim(
            model=model_total_hosps_latest,
            observed=observed_total_hosps,
            log=log.bind(note="via_intralevel_icu_shim"),
        )
        # total_hosp_shim is how much we shim the combined acute and icu data. We can apportion that
        # as a function of the relative weight. NB: If there is no total hospitalization data, then
        # total_hosp_shim will be 0 and this icu shim will also be 0. So this handles the check of
        # whether observed_total_hosps_latest is not None/np.nan in the strict shim function.
        model_icu_fraction = model_icu / model_total_hosps_latest
        shim = model_icu_fraction * total_hosp_shim
    elif observed_icu == 0:
        # As of 19 June 2020, the observed dataset is still being validated for erroneous inputs.
        # This includes cases of returning a 0 when we should be returning a np.nan/None.
        # For now, we will not apply a shim if the result returned is 0.
        shim = 0
    else:
        # We have ICU observed. In this case we will overwrite the natural model ratio of ICU to
        # total hospitalizations.
        shim = observed_icu - model_icu

    log.info(
        event="calculate_intralevel_icu_shim",
        shim=np.round(shim),
        observed=observed_icu,
        model=np.round(model_icu),
    )
    return shim
