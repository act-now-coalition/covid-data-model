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
        (0.0, 0.0, None, None),
        (0.0, None, 0.0, None),
        (None, 0.0, 0.0, None),
    ],
)
def test_calc_community_levels(
    weekly_cases_per_100k, beds_with_covid_ratio, weekly_admissions_per_100k, expected_level
):
    output = community_levels.calculate_community_level(
        weekly_cases_per_100k, beds_with_covid_ratio, weekly_admissions_per_100k
    )
    assert expected_level == output


def test_community_level_stale():
    fips = "36"
    dataset = build_one_region_dataset({CommonFields.CDC_COMMUNITY_LEVEL: [2] + [np.nan] * 16})
    metrics_df = top_level_metrics_test.build_metrics_df(
        fips,
        start_date="2020-12-01",
        weeklyNewCasesPer100k=[200] + [np.nan] * 15,
        # TODO(michael): Plumb in
        # weeklyAdmissionsPer100k=[0] * 16,
        bedsWithCovidPatientsRatio=[0] * 16,
    )

    results, latest = community_levels.calculate_community_level_timeseries_and_latest(
        dataset, metrics_df
    )

    # We allow cdc_community_level to be arbitrarily stale, so it should still be HIGH.
    assert latest.cdcCommunityLevel == CommunityLevel.HIGH
    assert latest.canCommunityLevel == None
