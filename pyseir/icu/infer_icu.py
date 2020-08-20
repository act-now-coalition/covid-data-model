import json
import structlog
import os.path

import pandas as pd

from pyseir import DATA_DIR
import libs.datasets.combined_datasets as combined_datasets

logger = structlog.get_logger()


class ICUConfig:
    WEIGHTS_PATHS = dict(
        population=os.path.join(DATA_DIR, "population_weights_via_fips.json"),
        one_month_trailing_cases=os.path.join(DATA_DIR, "one_month_trailing_weights_via_fips.json"),
    )
    LR_COEF = dict(m_hospitalized=0.2885, b=-1.6083)
    SUFFIX = "_superset"
    LOOKBACK_DAYS = 91
    LOOKBACK_DATE = pd.Timestamp.today() - pd.Timedelta(days=LOOKBACK_DAYS)


def get_data_for_icu_calc(fips: str) -> pd.DataFrame:
    """
    Get the timeseries data for the current aggregation level and the superset aggregation level.
    In the case where the current aggregation level is the highest (state), return the superset as
    state too.
    """
    COLUMNS = ["cases", "deaths", "current_icu", "current_hospitalized"]
    f = combined_datasets.get_timeseries_for_fips
    this_level_df = (
        f(fips, columns=COLUMNS).get_subset(after=ICUConfig.LOOKBACK_DATE).data.set_index("date")
    )
    super_level_df = (
        f(fips[:2], columns=COLUMNS)
        .get_subset(after=ICUConfig.LOOKBACK_DATE)
        .data.set_index("date")
    )
    return pd.merge(
        this_level_df,
        super_level_df,
        right_index=True,
        left_index=True,
        suffixes=("", ICUConfig.SUFFIX),
    )


def get_icu_timeseries(
    fips: str, use_actuals: bool = True, weight_by: str = "population"
) -> pd.Series:
    """
    """
    log = logger.new(fips=fips, event=f"ICU for Fips = {fips}")
    df = get_data_for_icu_calc(fips)
    has = df.apply(lambda x: not x.dropna().empty).to_dict()

    if use_actuals and has["current_icu"]:
        log.info(current_icu=True)
        return df["current_icu"]
    elif has["current_hospitalized"]:
        log.info(current_hosp=True)
        return estimate_icu_from_hospitalized(df["current_hospitalized"])
    else:
        # Get Superset ICU Timeseries
        if use_actuals and has["current_icu_superset"]:
            superset_icu = df["current_icu_superset"]
        else:
            # For now we are guaranteed that the superset has at least current_hospitalized
            # since all states have current_hospitalized. If we add another intermediate level, then
            # this logic will have to be changed.
            superset_icu = estimate_icu_from_hospitalized(df["current_hospitalized_superset"])

        # Get Disaggregation Weighting
        weight = get_weight_by_fips(fips, method=weight_by)
        log.info(disaggregation=True)
        return weight * superset_icu


def estimate_icu_from_hospitalized(current_hospitalized: pd.Series) -> pd.Series:
    """"""
    m = ICUConfig.LR_COEF["m_hospitalized"]
    b = ICUConfig.LR_COEF["b"]
    estimated_icu = m * current_hospitalized + b
    estimated_icu = estimated_icu.clip(lower=0)
    estimated_icu.name = "current_icu"
    return estimated_icu


def get_weight_by_fips(fips: str, method="population") -> float:
    """


    Currently Supported Methods
    population
        Distribute unattributed covid icu patients based on county population
    one_month_trailing_cases
        Distribute unattributed covid icu patients based on a county's fraction of the last month's
        total cases for that state.

    """
    if method in ICUConfig.WEIGHTS_PATHS:
        with open(ICUConfig.WEIGHTS_PATHS[method]) as f:
            weights = json.load(f)
    else:
        raise NotImplementedError
    return weights[fips]
