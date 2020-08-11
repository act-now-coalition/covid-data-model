import json
import structlog
import os.path

import pandas as pd

from pyseir import DATA_DIR
import libs.datasets.combined_datasets as combined_datasets

logger = structlog.get_logger()


class ICUConfig:
    WEIGHTS_PATHS = dict(population=os.path.join(DATA_DIR, "population_weights_via_fips.json"))
    LR_COEF = dict(m_hospitalized=0.275, b=-2.5)
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


def get_icu_timeseries(fips: str) -> pd.Series:
    """
    """
    log = logger.new(fips=fips, event=f"ICU for Fips = {fips}")
    df = get_data_for_icu_calc(fips)
    is_empty = df.apply(lambda x: x.dropna().empty).to_dict()

    if not is_empty["current_icu"]:
        log.info(current_icu=True)
        return df["current_icu"]
    else:
        log = log.bind(current_icu=False)

    if not is_empty["current_icu_superset"]:
        log.info(super_icu=True)
        if len(fips) == 2:
            return df["current_icu_superset"]
        else:
            weight = get_current_weight(fips)
            return weight * df["current_icu_superset"]
    else:
        log = log.bind(super_icu=False)

    estimated_icu_superset = estimate_icu_from_hospitalized(df["current_hospitalized_superset"])

    if len(fips) == 2:
        log.info(estimate_icu=True, disaggregate=False)
        return estimated_icu_superset
    else:
        log.info(estimate_icu=True, disaggregate=True)
        weight = get_current_weight(fips)
        return weight * estimated_icu_superset


def estimate_icu_from_hospitalized(current_hospitalized: pd.Series) -> pd.Series:
    """"""
    m = ICUConfig.LR_COEF["m_hospitalized"]
    b = ICUConfig.LR_COEF["b"]
    estimated_icu = m * current_hospitalized + b
    estimated_icu = estimated_icu.clip(lower=0)
    return estimated_icu


def get_current_weight(fips: str, method="population") -> float:
    if method == "population":
        with open(ICUConfig.WEIGHTS_PATHS[method]) as f:
            weights = json.load(f)
    else:
        raise NotImplementedError
    return weights[fips]
