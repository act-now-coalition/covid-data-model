from typing import List, Optional
import math
import numpy as np
import pandas as pd

from datapublic.common_fields import CommonFields
from libs.metrics.top_level_metrics import MetricsFields
from libs.metrics import top_level_metrics
from api import can_api_v2_definition

RiskLevel = can_api_v2_definition.RiskLevel


def calc_risk_level(value: Optional[float], thresholds: List[float]) -> RiskLevel:
    """Check the value against thresholds to determine the risk level for the metric.

    Each threshold is the upper limit for the risk level.

    Args:
        value: value of the metric.
        thresholds: list of values corresponding to thresholds for different risk levels

    Returns:
        A RiskLevel.
    """
    assert len(thresholds) in [3, 4], "Must pass low, med and high thresholds."
    if len(thresholds) == 3:
        level_low, level_med, level_high = thresholds
        level_critical = math.inf
    else:
        level_low, level_med, level_high, level_critical = thresholds
    if value is None:
        return RiskLevel.UNKNOWN

    if math.isinf(value) or np.isnan(value):
        return RiskLevel.UNKNOWN

    if value <= level_low:
        return RiskLevel.LOW
    elif value <= level_med:
        return RiskLevel.MEDIUM
    elif value <= level_high:
        return RiskLevel.HIGH
    elif value <= level_critical:
        return RiskLevel.CRITICAL

    return RiskLevel.EXTREME


def case_density_risk_level(value: float) -> RiskLevel:
    thresholds = [1, 10, 25, 75]
    return calc_risk_level(value, thresholds)


def test_positivity_risk_level(value: float) -> RiskLevel:
    thresholds = [0.03, 0.1, 0.2]
    return calc_risk_level(value, thresholds)


def contact_tracing_risk_level(value: float) -> RiskLevel:
    # Contact tracing risk level is flipped (lower value indicates a higher risk).
    level_high, level_med, level_low = (0, 0.1, 0.9)

    if value is None:
        return RiskLevel.UNKNOWN

    if value > level_low:
        return RiskLevel.LOW
    elif value > level_med:
        return RiskLevel.MEDIUM
    elif value >= level_high:
        return RiskLevel.HIGH

    return RiskLevel.UNKNOWN


def icu_capacity_ratio_risk_level(value: float) -> RiskLevel:
    thresholds = [0.7, 0.8, 0.85]
    return calc_risk_level(value, thresholds)


def infection_rate_risk_level(value: float) -> RiskLevel:
    thresholds = [0.9, 1.1, 1.4]
    return calc_risk_level(value, thresholds)


def top_level_risk_level(
    case_density_level: RiskLevel,
    test_positivity_level: RiskLevel,
    infection_rate_level: RiskLevel,
) -> RiskLevel:
    """Calculate the overall risk for a region based on metric risk levels."""
    if case_density_level in (RiskLevel.LOW, RiskLevel.UNKNOWN):
        return case_density_level

    level_list = [
        infection_rate_level,
        test_positivity_level,
        case_density_level,
    ]

    if RiskLevel.EXTREME in level_list:
        return RiskLevel.EXTREME
    elif RiskLevel.CRITICAL in level_list:
        return RiskLevel.CRITICAL
    elif RiskLevel.HIGH in level_list:
        return RiskLevel.HIGH
    elif RiskLevel.MEDIUM in level_list:
        return RiskLevel.MEDIUM
    elif RiskLevel.UNKNOWN in level_list:
        return RiskLevel.UNKNOWN

    return RiskLevel.LOW


def calculate_risk_level_from_metrics(
    metrics: can_api_v2_definition.Metrics,
) -> can_api_v2_definition.RiskLevels:
    case_density_level = case_density_risk_level(metrics.caseDensity)
    test_positivity_level = test_positivity_risk_level(metrics.testPositivityRatio)
    contact_tracing_level = contact_tracing_risk_level(metrics.contactTracerCapacityRatio)
    infection_rate_level = infection_rate_risk_level(metrics.infectionRate)
    icu_capacity_ratio_level = icu_capacity_ratio_risk_level(metrics.icuCapacityRatio)

    overall_level = top_level_risk_level(
        case_density_level, test_positivity_level, infection_rate_level,
    )
    levels = can_api_v2_definition.RiskLevels(
        overall=overall_level,
        testPositivityRatio=test_positivity_level,
        caseDensity=case_density_level,
        contactTracerCapacityRatio=contact_tracing_level,
        infectionRate=infection_rate_level,
        icuCapacityRatio=icu_capacity_ratio_level,
    )
    return levels


def calculate_risk_level_from_row(row: pd.Series):

    case_density_level = case_density_risk_level(row[MetricsFields.CASE_DENSITY_RATIO])
    test_positivity_level = test_positivity_risk_level(row[MetricsFields.TEST_POSITIVITY])
    infection_rate_level = infection_rate_risk_level(row[MetricsFields.INFECTION_RATE])

    return top_level_risk_level(case_density_level, test_positivity_level, infection_rate_level)


def calculate_risk_level_timeseries(
    metrics_df: pd.DataFrame, metric_max_lookback=top_level_metrics.MAX_METRIC_LOOKBACK_DAYS
):
    metrics_df = metrics_df.copy()
    # Shift infection rate to align with infection rate delay
    infection_rate = metrics_df[MetricsFields.INFECTION_RATE].shift(
        top_level_metrics.RT_TRUNCATION_DAYS
    )
    metrics_df[MetricsFields.INFECTION_RATE] = infection_rate
    metrics_df = metrics_df.set_index([CommonFields.DATE, CommonFields.FIPS])

    # We use the last available data within `MAX_METRIC_LOOKBACK_DAYS` to cacluate
    # the risk. Propagate the last value forward that many days so calculation for a given
    # day is the same as `top_level_metrics.calculate_latest_metrics`
    metrics_df.ffill(limit=metric_max_lookback - 1, inplace=True)

    overall_risk = metrics_df.apply(calculate_risk_level_from_row, axis=1)
    case_density_risk = metrics_df.apply(
        lambda row: case_density_risk_level(row[MetricsFields.CASE_DENSITY_RATIO]), axis=1
    )

    return pd.DataFrame(
        {"overall": overall_risk, MetricsFields.CASE_DENSITY_RATIO: case_density_risk}
    ).reset_index()
