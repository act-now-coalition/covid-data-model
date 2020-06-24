import numpy as np


def strict_shim(model: float, observed: float, log) -> float:
    """
    Take model outputs and calculate a "shim" value that can be added to the entire modeled
    cumulative deaths series to make them match the latest (i.e. today's) actual cumulative deaths.
    As a consequence, this will shift the future values by the same amount.

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

    log.info(event="strict_shim", shim=np.round(shim), observed=observed, model=np.round(model))
    return shim


def intralevel_icu_shim(
    model_acute_latest: float,
    model_icu_latest: float,
    observed_icu_latest: float,
    observed_total_hosps_latest: float,
    log,
):
    """
    Take model outputs and calculate a "shim" value that can be added to the icu series to make it
    match the latest (i.e. today's) actual icu.

    Parameters
    ----------
    model_acute_latest
        model estimate for acute hospitalized for latest date
    model_icu_latest
        model estimate for icu hospitalized for latest date
    observed_icu_latest
        observed for icu hospitalized for latest date
    observed_total_hosps_latest
        observed for total hospitalized for latest date
    log
        Log Instance
    Return
    ------
    shim: float
        Value to shim icu data by
    """
    # There are inconsistent None/"NaN" -> force all to np.nan for this scope.
    if observed_icu_latest is None:
        observed_icu_latest = np.nan

    if np.isnan(observed_icu_latest):
        # We don't have ICU specific data but let's try to have a shim informed by the shim used
        # for total hospitalization. This will maintain the same icu/total_hosp ratio that was
        # originally in the model.
        model_total_hosps_latest = model_acute_latest + model_icu_latest
        total_hosp_shim = strict_shim(
            model=model_total_hosps_latest,
            observed=observed_total_hosps_latest,
            log=log.bind(note="via_intralevel_icu_shim"),
        )
        # total_hosp_shim is how much we shim the combined acute and icu data. We can apportion that
        # as a function of the relative weight. NB: If there is no total hospitalization data, then
        # total_hosp_shim will be 0 and this icu shim will also be 0. So this handles the check of
        # whether observed_total_hosps_latest is not None/np.nan in the strict shim function.
        model_icu_fraction = model_icu_latest / model_total_hosps_latest
        shim = model_icu_fraction * total_hosp_shim
    elif observed_icu_latest == 0:
        # As of 19 June 2020, the observed dataset is still being validated for erroneous inputs.
        # This includes cases of returning a 0 when we should be returning a np.nan/None.
        # For now, we will not apply a shim if the result returned is 0.
        shim = 0
    else:
        # We have ICU observed. In this case we will overwrite the natural model ratio of ICU to
        # total hospitalizations.
        shim = observed_icu_latest - model_icu_latest

    log.info(
        event="intralevel_icu_shim",
        shim=np.round(shim),
        observed=observed_icu_latest,
        model=np.round(model_icu_latest),
    )
    return shim
