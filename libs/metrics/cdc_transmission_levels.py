from typing import List, Optional
import math
import numpy as np
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.metrics.top_level_metrics import MetricsFields
from libs.metrics import top_level_metrics
from api import can_api_v2_definition

TransmissionLevel = can_api_v2_definition.CDCTransmissionLevel


def calc_transmission_level(value: Optional[float], thresholds: List[float]) -> TransmissionLevel:
    """Check the value against thresholds to determine the transmission level for the metric.

    Each threshold is the upper limit for the risk level.

    Args:
        value: value of the metric.
        thresholds: list of values corresponding to thresholds for different risk levels

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
    thresholds = [10 / 7.0, 50 / 7.0, 100 / 7.0]
    return calc_transmission_level(value, thresholds)


def test_positivity_transmission_level(value: float) -> TransmissionLevel:
    thresholds = [0.05, 0.08, 0.10]
    return calc_transmission_level(value, thresholds)


def top_level_transmission_level(
    case_density_level: TransmissionLevel, test_positivity_level: TransmissionLevel,
) -> TransmissionLevel:
    """Calculate the overall risk for a region based on metric risk levels."""

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

    return RiskLevel.LOW


def calculate_transmission_level_from_metrics(
    metrics: can_api_v2_definition.Metrics,
) -> can_api_v2_definition.CDCTransmissionLevel:
    case_density_level = case_density_transmission_level(metrics.caseDensity)
    test_positivity_level = test_positivity_transmission_level(metrics.testPositivityRatio)

    overall_level = top_level_transmission_level(case_density_level, test_positivity_level)
    return overall_level


def calculate_transmission_level_from_row(row: pd.Series):

    case_density_level = case_density_transmission_level(row[MetricsFields.CASE_DENSITY_RATIO])
    test_positivity_level = test_positivity_transmission_level(row[MetricsFields.TEST_POSITIVITY])

    return top_level_transmission_level(case_density_level, test_positivity_level)


def calculate_transmission_level_timeseries(
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
    # the transmission. Propagate the last value forward that many days so calculation for a given
    # day is the same as `top_level_metrics.calculate_latest_metrics`
    metrics_df.ffill(limit=metric_max_lookback - 1, inplace=True)

    overall_transmission = metrics_df.apply(calculate_transmission_level_from_row, axis=1)

    return pd.DataFrame({"overall": overall_transmission,}).reset_index()
