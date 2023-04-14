from datapublic.common_fields import CommonFields
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.metrics.top_level_metrics import MAX_METRIC_LOOKBACK_DAYS, MetricsFields
from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd

from api import can_api_v2_definition

CommunityLevel = can_api_v2_definition.CommunityLevel
CommunityLevels = can_api_v2_definition.CommunityLevels


def is_invalid_value(value: Optional[float]) -> bool:
    # TODO(michael): Do we really need all these checks?
    return value is None or math.isinf(value) or np.isnan(value)


def calculate_community_level(
    weekly_cases_per_100k, beds_with_covid_ratio, weekly_admissions_per_100k
) -> Optional[CommunityLevel]:
    """Calculate the overall community level for a region based on metric levels.
    
    Note: As of April 14, 2023 we are harmonizing our score with the CDC in the
    case where Daily New Cases are missing, we are going to treat it as "low".

    If the number of cases in 7 days for a jurisdiction is missing, the
    7-day case rate is assigned to the “low” category. If both 7-day
    admissions and 7-day percentage inpatient beds indicators are N/A, the
    community burden category is assigned N/A.    

    We know have two states (Iowa and Florida) that have stopped case reporting.
    Instead of leaving them permanently gray on the map, we are planning to
    transition to flagging DNC as unknown, but still scoring on the hospital
    metrics.

    Therefore, an Unknown/None value for DNC no longer necessarily signals a
    data quality issue that we should address immediately.
    """

    # Return Unknown if no valid component metrics
    if (
        is_invalid_value(weekly_cases_per_100k)
        and is_invalid_value(beds_with_covid_ratio)
        and is_invalid_value(weekly_admissions_per_100k)
    ):
        return None

    weekly_cases_per_100k = weekly_cases_per_100k or 0
    beds_with_covid_ratio = beds_with_covid_ratio or 0
    weekly_admissions_per_100k = weekly_admissions_per_100k or 0

    if weekly_cases_per_100k < 200:
        if weekly_admissions_per_100k < 10 and beds_with_covid_ratio < 0.1:
            return CommunityLevel.LOW
        elif weekly_admissions_per_100k >= 20 or beds_with_covid_ratio >= 0.15:
            return CommunityLevel.HIGH
        else:
            return CommunityLevel.MEDIUM
    else:
        if weekly_admissions_per_100k < 10 and beds_with_covid_ratio < 0.1:
            return CommunityLevel.MEDIUM
        else:
            return CommunityLevel.HIGH


def calculate_community_level_from_metrics(
    metrics: can_api_v2_definition.Metrics,
) -> CommunityLevel:
    return calculate_community_level(
        metrics.weeklyNewCasesPer100k,
        metrics.bedsWithCovidPatientsRatio,
        metrics.weeklyCovidAdmissionsPer100k,
    )


def calculate_community_level_from_row(row: pd.Series):
    weekly_cases_per_100k = row[MetricsFields.WEEKLY_CASE_DENSITY_RATIO]
    beds_with_covid_ratio = row[MetricsFields.BEDS_WITH_COVID_PATIENTS_RATIO]
    weekly_covid_admissions_per_100k = row[MetricsFields.WEEKLY_COVID_ADMISSIONS_PER_100K]
    return calculate_community_level(
        weekly_cases_per_100k, beds_with_covid_ratio, weekly_covid_admissions_per_100k
    )


def calculate_community_level_timeseries_and_latest(
    timeseries: OneRegionTimeseriesDataset, metrics_df: pd.DataFrame
) -> Tuple[pd.DataFrame, CommunityLevels]:

    # Calculate CAN Community Levels from metrics. We allow canCommunityLevel to be based on
    # metrics that are up to MAX_METRIC_LOOKBACK_DAYS days old so ffill() them.
    metrics_df = metrics_df.set_index([CommonFields.DATE]).ffill(limit=MAX_METRIC_LOOKBACK_DAYS - 1)
    can_community_level_series = metrics_df.apply(calculate_community_level_from_row, axis=1)

    # Extract CDC Community Levels from raw timeseries.
    timeseries_df = timeseries.data.set_index([CommonFields.DATE])
    if CommonFields.CDC_COMMUNITY_LEVEL in timeseries_df.columns:
        cdc_community_level_series = timeseries_df[CommonFields.CDC_COMMUNITY_LEVEL].apply(
            lambda level: None if math.isnan(level) else CommunityLevel(level)
        )
    else:
        # CDC doesn't include data for all counties (e.g. DC county, some territories), so
        # create empty timeseries.
        index = timeseries_df.index
        cdc_community_level_series = pd.Series(data=[None] * index.size, index=index)

    community_levels_df = pd.DataFrame(
        {
            "canCommunityLevel": can_community_level_series,
            "cdcCommunityLevel": cdc_community_level_series,
        }
    ).reset_index()

    # Calculate latest CommunityLevels.
    # I suspect CDC will use the latest value no matter how stale it is. For
    # can_community_level we can just use the latest value since we did an
    # ffill() on the underlying metrics above.
    latest_cdc_community_level = cdc_community_level_series.ffill().iloc[-1]
    latest_can_community_level = can_community_level_series.iloc[-1]

    latest_community_levels = CommunityLevels(
        cdcCommunityLevel=latest_cdc_community_level, canCommunityLevel=latest_can_community_level
    )

    return community_levels_df, latest_community_levels
