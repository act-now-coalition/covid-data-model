from typing import Sequence

import numpy as np


def shim_deaths_model_to_observations(
    model_death_ts: Sequence, idx: int, observed_latest: dict, log
):
    """
    Take model outputs and calculate a "shim" value that can be added to the entire modeled
    cumulative deaths series to make them match the latest (i.e. today's) actual cumulative deaths.
    As a consequence, this will shift the future values by the same amount.

    Parameters
    ----------
    model_death_ts
        Model array sequence for cumulative deaths
    idx
        Index on which to align
    observed_latest
        Dictionary of latest values taken from combined dataset for the fips of interest.
    log
        Log Instance
    Return
    ------
    shimmed_death: float
        Value to shim the timeseries by
    """
    observed_latest_cum_deaths = observed_latest["deaths"]

    # There are inconsistent None/"NaN" -> force all to np.nan for this scope.
    if observed_latest_cum_deaths is None:
        observed_latest_cum_deaths = np.nan

    model_latest_cum_deaths = model_death_ts[idx]

    if np.isnan(observed_latest_cum_deaths):
        death_shim = 0
    elif observed_latest_cum_deaths == 0:
        # As of 19 June 2020, the observed dataset is still being validated for erroneous inputs.
        # This includes cases of returning a 0 when we should be returning a np.nan/None.
        # For now, we will not apply a shim if the result returned is 0.
        death_shim = 0
    else:
        death_shim = observed_latest_cum_deaths - model_latest_cum_deaths

    log.info(
        event="Death Shim Applied:",
        death_shim=np.round(death_shim),
        observed_latest_cum_deaths=np.round(observed_latest_cum_deaths),
        model_latest_cum_deaths=np.round(model_latest_cum_deaths),
    )

    return death_shim
