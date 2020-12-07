import pytest
from libs.top_level_metric_risk_levels import RiskLevel
from libs import top_level_metric_risk_levels


def test_calc_risk_level_below_limit():
    """
    Value in between two risk levels should take the higher level.
    """
    assert top_level_metric_risk_levels.calc_risk_level(2, [1, 3, 4]) == RiskLevel.MEDIUM


def test_calc_risk_level_critical():
    assert top_level_metric_risk_levels.calc_risk_level(5, [1, 3, 4]) == RiskLevel.CRITICAL


def test_calc_risk_level_at_limit():
    """
    Value at upper limit of a risk level should equal that risk level.
    """
    assert top_level_metric_risk_levels.calc_risk_level(1, [1, 3, 4]) == RiskLevel.LOW


def test_calc_risk_level_extreme():
    assert top_level_metric_risk_levels.calc_risk_level(6, [1, 3, 4, 5]) == RiskLevel.EXTREME


@pytest.mark.parametrize(
    "value,expected_level", [(1.0, RiskLevel.LOW), (0.9, RiskLevel.MEDIUM), (0.0, RiskLevel.HIGH)],
)
def test_contact_tracing_levels(value, expected_level):
    assert top_level_metric_risk_levels.contact_tracing_risk_level(value) == expected_level


@pytest.mark.parametrize(
    "case_density_level,expected",
    [
        (RiskLevel.LOW, RiskLevel.LOW),
        (RiskLevel.MEDIUM, RiskLevel.CRITICAL),
        (RiskLevel.EXTREME, RiskLevel.EXTREME),
    ],
)
def test_top_level_risk_level_case_density_override(case_density_level, expected):
    top_level_risk = top_level_metric_risk_levels.top_level_risk_level(
        case_density_level=expected,
        test_positivity_level=RiskLevel.CRITICAL,
        contact_tracing_level=RiskLevel.MEDIUM,
        icu_headroom_level=RiskLevel.MEDIUM,
        infection_rate_level=RiskLevel.LOW,
    )

    assert top_level_risk == expected
