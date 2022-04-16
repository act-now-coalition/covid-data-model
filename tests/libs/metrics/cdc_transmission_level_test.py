import pandas as pd
import pytest
from api.can_api_v2_definition import CDCTransmissionLevel, Metrics
from libs.metrics.cdc_transmission_levels import TransmissionLevel
from libs.metrics import cdc_transmission_levels
from tests.libs.metrics import top_level_metrics_test


def test_calc_transmission_level_below_limit():
    """
    Value in between two transmission levels should take the higher level.
    """
    assert (
        cdc_transmission_levels.calc_transmission_level(2, [1, 3, 4]) == TransmissionLevel.MODERATE
    )


def test_calc_transmission_level_high():
    assert cdc_transmission_levels.calc_transmission_level(5, [1, 3, 4]) == TransmissionLevel.HIGH


def test_calc_transmission_level_at_limit():
    # Value at upper limit of a transmission level should equal the transmission level above
    assert (
        cdc_transmission_levels.calc_transmission_level(1, [1, 3, 4]) == TransmissionLevel.MODERATE
    )


@pytest.mark.parametrize(
    "test_positivity,case_density,expected_level",
    [
        (0.0, 9.99 / 7, TransmissionLevel.LOW),
        (0.10, 0.0 / 7, TransmissionLevel.HIGH),
        (0.0, 100.0 / 7, TransmissionLevel.HIGH),
        (0.0, 99.9 / 7, TransmissionLevel.SUBSTANTIAL),
        (0.0, None, TransmissionLevel.LOW),
        (None, None, TransmissionLevel.UNKNOWN),
    ],
)
def test_calc_transmission_levels(test_positivity, case_density, expected_level):
    metrics = Metrics.empty()
    metrics.testPositivityRatio = test_positivity
    metrics.caseDensity = case_density
    output = cdc_transmission_levels.calculate_transmission_level_from_metrics(metrics)
    assert expected_level == output


def test_transmission_level_timeseries():
    fips = "36"
    metrics_df = top_level_metrics_test.build_metrics_df(
        fips,
        start_date="2020-12-01",
        caseDensity=[7.0] * 16,
        testPositivityRatio=[0.8] * 1 + [None] * 15,
    )

    results = cdc_transmission_levels.calculate_transmission_level_timeseries(metrics_df)

    # Last day is expected to be MODERATE because test positivity is stale
    expected_cdc_transmission_level = [CDCTransmissionLevel.HIGH] * 15 + [
        CDCTransmissionLevel.MODERATE
    ]
    expected_cdc_transmission_level = pd.Series(
        expected_cdc_transmission_level, name="cdcTransmissionLevel"
    )

    pd.testing.assert_series_equal(
        results.loc[:, "cdcTransmissionLevel"], expected_cdc_transmission_level
    )
