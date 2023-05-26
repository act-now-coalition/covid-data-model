import numpy as np
import pytest
from api.can_api_v2_definition import CommunityLevel
from datapublic.common_fields import CommonFields
from libs.metrics import community_levels
from tests.libs.metrics import top_level_metrics_test
from tests.test_helpers import build_one_region_dataset


@pytest.mark.parametrize(
    "weekly_cases_per_100k, beds_with_covid_ratio, weekly_admissions_per_100k, expected_level",
    [
        (199, 0.0999, 9.99, CommunityLevel.LOW),
        (199, 0.10, 9.99, CommunityLevel.MEDIUM),
        (199, 0.0999, 10, CommunityLevel.MEDIUM),
        (199, 0.149, 19.99, CommunityLevel.MEDIUM),
        (199, 0.15, 9.99, CommunityLevel.HIGH),
        (199, 0.0999, 20, CommunityLevel.HIGH),
        (200, 0.0999, 9.99, CommunityLevel.MEDIUM),
        (200, 0.10, 9.99, CommunityLevel.HIGH),
        (200, 0.0999, 10, CommunityLevel.HIGH),
        (0.0, 0.0, None, CommunityLevel.LOW),
        (0.0, None, 0.0, CommunityLevel.LOW),
        (None, 0.0, 0.0, CommunityLevel.LOW),  # 14 April 2023: No DNC now valid.
        (None, None, 0.0, CommunityLevel.LOW),
        (None, None, None, None),
    ],
)
def test_calc_community_levels(
    weekly_cases_per_100k, beds_with_covid_ratio, weekly_admissions_per_100k, expected_level
):
    output = community_levels.calculate_community_level(
        weekly_cases_per_100k, beds_with_covid_ratio, weekly_admissions_per_100k
    )
    assert expected_level == output


@pytest.mark.parametrize(
    "days_stale, expected_community_level",
    [
        (1, CommunityLevel.MEDIUM),
        (13, CommunityLevel.MEDIUM),
        (14, CommunityLevel.MEDIUM),
        (15, CommunityLevel.MEDIUM),
        (16, CommunityLevel.LOW),
        (100, CommunityLevel.LOW),
    ],
)
def test_stale_community_levels(days_stale, expected_community_level):
    fips = "36"
    # Valid cdcCommunityLevel x days back, followed by nans.
    dataset = build_one_region_dataset(
        {CommonFields.CDC_COMMUNITY_LEVEL: [0] + [np.nan] * (days_stale - 1)}
    )

    # No valid case data for x days.
    metrics_df = top_level_metrics_test.build_metrics_df(
        fips,
        start_date="2020-12-01",
        weeklyNewCasesPer100k=[200] + [np.nan] * (days_stale - 1),
        weeklyAdmissionsPer100k=[0] * days_stale,
        bedsWithCovidPatientsRatio=[0] * days_stale,
    )

    _, latest = community_levels.calculate_community_level_timeseries_and_latest(
        dataset, metrics_df
    )

    # We allow cdcCommunityLevel to be arbitrarily stale, so it should still be LOW.
    # Note this isn't parameterized, but just is returned in the same fn call
    assert latest.cdcCommunityLevel == CommunityLevel.LOW
    # canCommunityLevel is missing case data for 15 days, so it will be None.
    # As of 14 April 2023, we now are accepting None, so it should be low.
    assert latest.canCommunityLevel == expected_community_level
