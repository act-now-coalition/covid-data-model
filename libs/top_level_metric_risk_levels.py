import enum
from typing import List

from covidactnow.datapublic import common_fields


class RiskLevel(common_fields.ValueAsStrMixin, str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


def calc_risk_level(value: float, thresholds: List[float]):
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
