import enum
from typing import List, Optional

from covidactnow.datapublic import common_fields
from api import can_api_v2_definition

RiskLevel = can_api_v2_definition.RiskLevel


def calc_risk_level(value: Optional[float], thresholds: List[float]):
    """
    Check the value against thresholds to determine the risk level for the metric. Each threshold is the
    upper limit for the risk level.

    Args:
        value: value of the metric.
        thresholds: list of values corresponding to thresholds for different risk levels

    Returns:
        A RiskLevel.
    """
    assert len(thresholds) == 3, "Must pass low, med and high thresholds."
    level_low, level_med, level_high = thresholds

    if value is None:
        return RiskLevel.UNKNOWN

    risk_level = RiskLevel.UNKNOWN
    if value <= level_low:
        risk_level = RiskLevel.LOW
    elif value <= level_med:
        risk_level = RiskLevel.MEDIUM
    elif value <= level_high:
        risk_level = RiskLevel.HIGH
    elif value > level_high:
        risk_level = RiskLevel.CRITICAL
    return risk_level


def case_density_risk_level(value: float):
    thresholds = [1, 10, 25]
    return calc_risk_level(value, thresholds)


def test_positivity_risk_level(value: float):
    thresholds = [0.03, 0.1, 0.2]
    return calc_risk_level(value, thresholds)


def contact_tracing_risk_level(value: float):
    # NOTE: Setting the upperLimit to 0 means we will not grade anybody as
    # RiskLevel.LOW ("Critical" on the website). The lowest grade you can get is
    # RiskLevel.MEDIUM.
    thresholds = [0, 0.1, 0.9]
    return calc_risk_level(value, thresholds)


def icu_headroom_ratio_risk_level(value: float):
    thresholds = [0.5, 0.6, 0.7]
    return calc_risk_level(value, thresholds)


def infection_rate_risk_level(value: float):
    thresholds = [0.9, 1.1, 1.4]
    return calc_risk_level(value, thresholds)


def top_level_risk_level(
    case_density_level: RiskLevel,
    test_positivity_level: RiskLevel,
    contact_tracing_level: RiskLevel,
    icu_headroom_level: RiskLevel,
    infection_rate_level: RiskLevel,
):
    """
    Calculate the overall risk for a region based on metric risk levels.
    """
    levelList = [
        infection_rate_level,
        icu_headroom_level,
        test_positivity_level,
        case_density_level,
    ]
    top_level_risk = RiskLevel.UNKNOWN

    if RiskLevel.CRITICAL in levelList or contact_tracing_level == RiskLevel.LOW:
        top_level_risk = RiskLevel.CRITICAL
    elif RiskLevel.HIGH in levelList or contact_tracing_level == RiskLevel.MEDIUM:
        top_level_risk = RiskLevel.HIGH
    elif RiskLevel.MEDIUM in levelList or contact_tracing_level == RiskLevel.HIGH:
        top_level_risk = RiskLevel.MEDIUM
    elif RiskLevel.UNKNOWN in levelList or contact_tracing_level == RiskLevel.UNKNOWN:
        top_level_risk = RiskLevel.UNKNOWN
    else:
        top_level_risk = RiskLevel.LOW
    return top_level_risk


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
    return can_api_v2_definition.RiskLevels(
        overall=overall_level,
        testPositivityRatio=test_positivity_level,
        caseDensity=case_density_level,
        contactTracerCapacityRatio=contact_tracing_level,
        infectionRate=infection_rate_level,
        icuHeadroomRatio=icu_headroom_level,
    )
