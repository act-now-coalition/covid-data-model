from typing import Sequence

import numpy as np


def shim_deaths_model_to_observations(
    model_death_ts: Sequence, idx: int, observed_latest: dict, log
):
    """Enforce the constraint that cumulative model deaths up to today equals cumulative observed
    deaths up to today. Take model outputs and shim them s.t. the latest observed value matches the
    same value for the outputted model.

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
        death_shim = 0
    else:
        # To be consistent with the other shims coming, I will shim all values by the same amount
        # so that the cumulative model today equals cumulative observed today. This will shift the
        # future values by the same amount.
        death_shim = observed_latest_cum_deaths - model_latest_cum_deaths

    log.info(
        event="Death Shim Applied:",
        death_shim=np.round(death_shim),
        observed_latest_cum_deaths=np.round(observed_latest_cum_deaths),
        model_latest_cum_deaths=np.round(model_latest_cum_deaths),
    )

    return death_shim
