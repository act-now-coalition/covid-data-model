from typing import List, Optional
import math
import numpy as np

from api import can_api_v2_definition

TransmissionLevel = can_api_v2_definition.CDCTransmissionLevel


def calc_transmission_level(value: Optional[float], thresholds: List[float]) -> TransmissionLevel:
    """Check the value against thresholds to determine the transmission level for the metric.

    Each threshold is the upper limit for the transmission level.

    Args:
        value: value of the metric.
        thresholds: list of values corresponding to thresholds for different transmission levels

    Returns:
        A CDCTranmissionLevel.
    """
    assert len(thresholds) == 3, "Must pass low, med and substantial thresholds."
    if len(thresholds) == 3:
        level_low, level_moderate, level_substantial = thresholds
        level_high = math.inf

    if value is None or math.isinf(value) or np.isnan(value):
        return TransmissionLevel.UNKNOWN

    if value < level_low:
        return TransmissionLevel.LOW
    elif value < level_moderate:
        return TransmissionLevel.MODERATE
    elif value < level_substantial:
        return TransmissionLevel.SUBSTANTIAL

    return TransmissionLevel.HIGH


def case_density_transmission_level(value: float) -> TransmissionLevel:
    # CDC thresholds are number of cases over a 7 day average. Our case density
    # metrics are cases over a 7 day average.  To square the two, we should divide
    # CDC thresholds by 7 to get the equivalent threshold for our 7-day averages.
    thresholds = [10 / 7.0, 50 / 7.0, 100 / 7.0]
    return calc_transmission_level(value, thresholds)


def test_positivity_transmission_level(value: float) -> TransmissionLevel:
    thresholds = [0.05, 0.08, 0.10]
    return calc_transmission_level(value, thresholds)


def overall_transmission_level(
    case_density_level: TransmissionLevel, test_positivity_level: TransmissionLevel,
) -> TransmissionLevel:
    """Calculate the overall transmission level for a region based on metric transmission levels.

    Args:
        case_density_level: Case density transmission level.
        test_positivity_level: Test positivity transmission level.
    """

    level_list = [
        test_positivity_level,
        case_density_level,
    ]

    if TransmissionLevel.HIGH in level_list:
        return TransmissionLevel.HIGH
    elif TransmissionLevel.SUBSTANTIAL in level_list:
        return TransmissionLevel.SUBSTANTIAL
    elif TransmissionLevel.MODERATE in level_list:
        return TransmissionLevel.MODERATE
    elif TransmissionLevel.UNKNOWN in level_list:
        return TransmissionLevel.UNKNOWN

    return TransmissionLevel.LOW


def calculate_transmission_level_from_metrics(
    metrics: can_api_v2_definition.Metrics,
) -> can_api_v2_definition.CDCTransmissionLevel:
    case_density_level = case_density_transmission_level(metrics.caseDensity)
    test_positivity_level = test_positivity_transmission_level(metrics.testPositivityRatio)

    overall_level = overall_transmission_level(case_density_level, test_positivity_level)
    return overall_level
