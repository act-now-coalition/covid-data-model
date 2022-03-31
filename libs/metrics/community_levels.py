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
    """Calculate the overall community level for a region based on metric levels."""

    if (
        is_invalid_value(weekly_cases_per_100k)
        or is_invalid_value(beds_with_covid_ratio)
        or is_invalid_value(weekly_admissions_per_100k)
    ):
        # TODO(michael): For now, return None if we are missing any of the
        # underlying metrics. Once we see how bad this is we can compare against
        # CDC and decide how to handle.
        return None

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

    # Calculate CAN Community Levels from metrics.
    metrics_df = metrics_df.set_index([CommonFields.DATE])
    can_community_level_series = metrics_df.apply(calculate_community_level_from_row, axis=1)

    # Extract CDC Community Levels from raw timeseries.
    timeseries_df = timeseries.data.set_index([CommonFields.DATE])
    cdc_community_level_series = timeseries_df[CommonFields.CDC_COMMUNITY_LEVEL].apply(
        lambda level: None if math.isnan(level) else CommunityLevel(level)
    )

    community_levels_df = pd.DataFrame(
        {
            "canCommunityLevel": can_community_level_series,
            "cdcCommunityLevel": cdc_community_level_series,
        }
    ).reset_index()

    # Calculate latest CommunityLevels.
    # I suspect CDC will use the latest value no matter how stale it is. For
    # can_community_level we only want to look back MAX_METRIC_LOOKBACK_DAYS.
    latest_cdc_community_level = cdc_community_level_series.ffill().iloc[-1]
    latest_can_community_level = can_community_level_series.ffill(
        limit=MAX_METRIC_LOOKBACK_DAYS - 1
    ).iloc[-1]

    latest_community_levels = CommunityLevels(
        cdcCommunityLevel=latest_cdc_community_level, canCommunityLevel=latest_can_community_level
    )

    return community_levels_df, latest_community_levels
