import math
from typing import List, Optional

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

    if math.isinf(value):
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


def icu_headroom_ratio_risk_level(value: float) -> RiskLevel:
    thresholds = [0.5, 0.6, 0.7]
    return calc_risk_level(value, thresholds)


def infection_rate_risk_level(value: float) -> RiskLevel:
    thresholds = [0.9, 1.1, 1.4]
    return calc_risk_level(value, thresholds)


def top_level_risk_level(
    case_density_level: RiskLevel,
    test_positivity_level: RiskLevel,
    contact_tracing_level: RiskLevel,
    icu_headroom_level: RiskLevel,
    infection_rate_level: RiskLevel,
) -> RiskLevel:
    """Calculate the overall risk for a region based on metric risk levels."""
    if case_density_level is RiskLevel.LOW:
        return RiskLevel.LOW

    level_list = [
        infection_rate_level,
        contact_tracing_level,
        icu_headroom_level,
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
    icu_headroom_level = icu_headroom_ratio_risk_level(metrics.icuHeadroomRatio)
    infection_rate_level = infection_rate_risk_level(metrics.infectionRate)

    overall_level = top_level_risk_level(
        case_density_level,
        test_positivity_level,
        contact_tracing_level,
        icu_headroom_level,
        infection_rate_level,
    )
    levels = can_api_v2_definition.RiskLevels(
        overall=overall_level,
        testPositivityRatio=test_positivity_level,
        caseDensity=case_density_level,
        contactTracerCapacityRatio=contact_tracing_level,
        infectionRate=infection_rate_level,
        icuHeadroomRatio=icu_headroom_level,
    )
    return levels
