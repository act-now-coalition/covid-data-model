import collections
import json
from functools import lru_cache
from typing import Mapping
from typing import Optional

import structlog
import os.path
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from libs import pipeline
from libs.datasets.timeseries import TimeseriesDataset
from pyseir import DATA_DIR
from covidactnow.datapublic.common_fields import CommonFields

logger = structlog.get_logger()


class ICUConfig:
    LOOKBACK_DATE = pd.Timestamp.today() - pd.Timedelta(days=91)


class ICUWeightsPath(Enum):
    # Distribute unattributed covid icu patients based on county population
    POPULATION = os.path.join(DATA_DIR, "population_weights_via_fips.json")
    # Distribute unattributed covid icu patients based on a county's fraction of the last month's
    # total cases for that state (pre-trained out of band).
    ONE_MONTH_TRAILING_CASES = os.path.join(DATA_DIR, "one_month_trailing_weights_via_fips.json")


@dataclass
class LinearRegressionCoefficients:
    m_hospitalized: float = 0.2885
    b: float = -1.6083


def get_icu_timeseries(
    region: pipeline.Region,
    regional_combined_data: TimeseriesDataset,
    state_combined_data: Optional[TimeseriesDataset],
    use_actuals: bool = True,
    weight_by: ICUWeightsPath = ICUWeightsPath.POPULATION,
) -> pd.Series:
    """
    Loads data for region of interest and returns ICU Utilization numerator.

    Args:
        region: Region of interest
        use_actuals: If True, return actuals when available. If False, always use predictions.
        weight_by: The method by which to estimate county level utilization from state level utilization when
            no county level inpatient/icu data is available.

    Returns:
        A date-indexed series of ICU estimate for heads-in-beds for a given region
    """
    log = logger.new(region=str(region), event=f"ICU for Region = {region}")
    data = _get_data_for_icu_calc(regional_combined_data, state_combined_data)
    return _calculate_icu_timeseries(
        data=data,
        region=region,
        use_actuals=use_actuals,
        block_current_icu_and_hospitilized=(region.fips == "36061"),
        weight_by=weight_by,
        log=log,
    )


def _calculate_icu_timeseries(
    data: pd.DataFrame,
    region: pipeline.Region,
    use_actuals: bool = True,
    block_current_icu_and_hospitilized: bool = False,
    weight_by: ICUWeightsPath = ICUWeightsPath.POPULATION,
    log=None,
) -> pd.Series:
    """
        Load data for region of interest and return ICU Utilization numerator.

    Parameters
    ----------
    data: pd.DataFrame
        DataFrame for the region of interest (which includes the data for the higher aggregation
        level of that region of interest.
    region: pipeline.Region
        Region of interest key used for calculating state-to-county disaggregation weights as needed
    use_actuals: bool
        If True, return actuals when available. If False, always use predictions.
    weight_by: ICUWeightsPath
        The method by which to estimate county level utilization from state level utilization when
        no county level inpatient/icu data is available.
    log:
        Logging object

    Returns
    -------
    output: pandas.Series
        A date-indexed series of ICU estimate for heads-in-beds for a given region
    """
    has = data.apply(lambda x: not x.dropna().empty).to_dict()

    if block_current_icu_and_hospitilized:
        has["current_icu"] = False
        has["current_hospitalized"] = False

    if use_actuals and has["current_icu"]:
        log.info(current_icu=True)
        return data["current_icu"]
    elif has["current_hospitalized"]:
        log.info(current_hosp=True)
        return _estimate_icu_from_hospitalized(data["current_hospitalized"])
    else:
        # Get Superset ICU Timeseries
        if use_actuals and has["current_icu_superset"]:
            superset_icu = data["current_icu_superset"]
        else:
            # For now we are guaranteed that the superset has at least current_hospitalized
            # since all states have current_hospitalized. If we add another intermediate level, then
            # this logic will have to be changed.
            superset_icu = _estimate_icu_from_hospitalized(data["current_hospitalized_superset"])

        # Get Disaggregation Weighting
        weight = _get_region_weight_map()[region][weight_by]
        log.info(disaggregation=True)
        return weight * superset_icu


def _get_data_for_icu_calc(
    regional_combined_data: TimeseriesDataset,
    state_combined_data: Optional[TimeseriesDataset],
    lookback_date=ICUConfig.LOOKBACK_DATE,
) -> pd.DataFrame:
    """
    Get the timeseries data for the current aggregation level and the superset aggregation level.
    In the case where the current aggregation level is the highest (state), return the superset as
    state too.

    Parameters
    ----------
    lookback_date: bool
        The start date for the returned estimate. The linear regression estimate is
        fit on recent data, with no assumption that the process is stationary. So the further back
        in time you go, the more uncertain the estimate.

    Returns
    -------
    output: pandas.DataFrame
        Dataframe with columns for the region of interest and that regions superset aggregation
        level. The column names for the superset are appended with "_superset"

    """
    COLUMNS = [
        CommonFields.CASES,
        CommonFields.DEATHS,
        CommonFields.CURRENT_ICU,
        CommonFields.CURRENT_HOSPITALIZED,
    ] + TimeseriesDataset.INDEX_FIELDS

    this_level_df = regional_combined_data.get_subset(
        after=lookback_date, columns=COLUMNS
    ).data.set_index("date")
    if state_combined_data:
        super_level_df = state_combined_data.get_subset(
            after=lookback_date, columns=COLUMNS
        ).data.set_index("date")
    else:
        super_level_df = this_level_df
    return pd.merge(
        this_level_df,
        super_level_df,
        right_index=True,
        left_index=True,
        suffixes=("", "_superset"),
    )


def _estimate_icu_from_hospitalized(
    current_hospitalized: pd.Series,
    coefficients: LinearRegressionCoefficients = LinearRegressionCoefficients(),
) -> pd.Series:
    """
    Parameters
    ----------
    current_hospitalized: pandas.Series
        Input sequence to be converted
    coefficients:
        Linear regression coefficients to be applied

    Returns
    -------
    Pandas Series transformed according to coefficients
    """
    m = coefficients.m_hospitalized
    b = coefficients.b
    estimated_icu = m * current_hospitalized + b
    estimated_icu = estimated_icu.clip(lower=0)
    estimated_icu.name = "current_icu"
    return estimated_icu


@lru_cache(None)
def _get_region_weight_map() -> Mapping[pipeline.Region, Mapping[ICUWeightsPath, float]]:
    """Returns a map of maps with region and icu weight paths as keys."""
    region_map = collections.defaultdict(dict)
    for method in [ICUWeightsPath.POPULATION, ICUWeightsPath.ONE_MONTH_TRAILING_CASES]:
        with open(method.value) as f:
            weights = json.load(f)
            for fips, weight in weights.items():
                region_map[pipeline.Region.from_fips(fips)][method] = weight
    return region_map
