from libs.top_level_metric_risk_levels import RiskLevel, calc_risk_level, top_level_risk_level


def test_calc_risk_level_below_limit():
    """
    Value in between two risk levels should take the higher level.
    """
    assert calc_risk_level(2, [1, 3, 4]) == RiskLevel.MEDIUM


def test_calc_risk_level_at_limit():
    """
    Value at upper limit of a risk level should equal that risk level.
    """
    assert calc_risk_level(1, [1, 3, 4]) == RiskLevel.LOW


def test_top_level_risk_level_critical():
    """
    Critical test_positivity tracing should be critical.
    """
    top_level_risk = top_level_risk_level(
        case_density_level=RiskLevel.LOW,
        test_positivity_level=RiskLevel.CRITICAL,
        contact_tracing_level=RiskLevel.MEDIUM,
        icu_headroom_level=RiskLevel.MEDIUM,
        infection_rate_level=RiskLevel.LOW,
    )

    assert top_level_risk == RiskLevel.CRITICAL


def test_top_level_risk_level_contact_tracing():
    """
    Low contact tracing should be critical.
    """
    top_level_risk = top_level_risk_level(
        case_density_level=RiskLevel.LOW,
        test_positivity_level=RiskLevel.MEDIUM,
        contact_tracing_level=RiskLevel.LOW,
        icu_headroom_level=RiskLevel.MEDIUM,
        infection_rate_level=RiskLevel.LOW,
    )

    assert top_level_risk == RiskLevel.CRITICAL
