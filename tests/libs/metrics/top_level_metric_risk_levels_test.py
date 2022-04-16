import pytest
import pandas as pd
from libs.metrics.top_level_metric_risk_levels import RiskLevel
from libs.metrics import top_level_metric_risk_levels as metric_risk_levels
from libs.metrics import top_level_metrics
from tests.libs.metrics import top_level_metrics_test


def test_calc_risk_level_below_limit():
    """
    Value in between two risk levels should take the higher level.
    """
    assert metric_risk_levels.calc_risk_level(2, [1, 3, 4]) == RiskLevel.MEDIUM


def test_calc_risk_level_critical():
    assert metric_risk_levels.calc_risk_level(5, [1, 3, 4]) == RiskLevel.CRITICAL


def test_calc_risk_level_at_limit():
    """
    Value at upper limit of a risk level should equal that risk level.
    """
    assert metric_risk_levels.calc_risk_level(1, [1, 3, 4]) == RiskLevel.LOW


def test_calc_risk_level_extreme():
    assert metric_risk_levels.calc_risk_level(6, [1, 3, 4, 5]) == RiskLevel.EXTREME


@pytest.mark.parametrize(
    "value,expected_level", [(1.0, RiskLevel.LOW), (0.9, RiskLevel.MEDIUM), (0.0, RiskLevel.HIGH)],
)
def test_contact_tracing_levels(value, expected_level):
    assert metric_risk_levels.contact_tracing_risk_level(value) == expected_level


@pytest.mark.parametrize(
    "case_density_level,expected",
    [
        (RiskLevel.UNKNOWN, RiskLevel.UNKNOWN),
        (RiskLevel.LOW, RiskLevel.LOW),
        (RiskLevel.MEDIUM, RiskLevel.CRITICAL),
        (RiskLevel.EXTREME, RiskLevel.EXTREME),
    ],
)
def test_top_level_risk_level_case_density_override(case_density_level, expected):
    top_level_risk = metric_risk_levels.top_level_risk_level(
        case_density_level=expected,
        test_positivity_level=RiskLevel.CRITICAL,
        infection_rate_level=RiskLevel.LOW,
    )

    assert top_level_risk == expected


def test_risk_level_timeseries():
    fips = "36"
    metrics_df = top_level_metrics_test.build_metrics_df(
        fips,
        start_date="2020-12-01",
        caseDensity=[7.0] * 16,
        testPositivityRatio=[0.8] * 1 + [None] * 15,
    )

    metrics = top_level_metrics.calculate_latest_metrics(metrics_df, None)
    expected_latest_risk_level = metric_risk_levels.calculate_risk_level_from_metrics(metrics)

    results = metric_risk_levels.calculate_risk_level_timeseries(metrics_df)

    # Last day is expected to be medium because test positivity is stale
    expected_overall = [RiskLevel.CRITICAL] * 15 + [RiskLevel.MEDIUM]
    expected_overall = pd.Series(expected_overall, name="overall")

    expected_case_density = [RiskLevel.MEDIUM] * 16
    expected_case_density = pd.Series(expected_case_density, name="caseDensity")

    pd.testing.assert_series_equal(results.loc[:, "caseDensity"], expected_case_density)
    pd.testing.assert_series_equal(results.loc[:, "overall"], expected_overall)

    assert expected_latest_risk_level.overall == expected_overall.iloc[-1]
