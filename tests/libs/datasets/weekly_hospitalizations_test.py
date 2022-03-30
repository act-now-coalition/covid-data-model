from libs.pipeline import Region
from libs.datasets import weekly_hospitalizations
from tests import test_helpers
from datapublic.common_fields import CommonFields
import pytest


def test_add_weekly_hospitalizations():
    data = {
        Region.from_fips("34001"): {
            CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID: [2, 3, 4, 2],
            CommonFields.NEW_CASES: [1, 2, 3, 4, 5],
        },
        Region.from_cbsa_code("11100"): {
            CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID: [3, 4, 5],
            CommonFields.NEW_CASES: [1, 2, 3, 4, 5],
        },
        Region.from_state("MA"): {
            CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID: [40, 50, 60, 50, 60, 40, 30, 40, 50, 70],
            CommonFields.NEW_CASES: [1, 2, 3, 4, 5],
        },
    }
    dataset_in = test_helpers.build_dataset(data)
    out = weekly_hospitalizations.add_weekly_hospitalizations(dataset_in=dataset_in)

    expected_data = {
        **data,
        Region.from_state("MA"): {
            CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID: [40, 50, 60, 50, 60, 40, 30, 40, 50, 70],
            CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID: [None] * 6 + [330] * 3 + [340],
            CommonFields.NEW_CASES: [1, 2, 3, 4, 5],
        },
    }
    expected = test_helpers.build_dataset(expected_data)
    test_helpers.assert_dataset_like(expected, out)


def test_add_weekly_hospitalizations_no_overwrite():
    # Test that we don't add weekly data if it already exists (should raise an error).
    dataset_in = test_helpers.build_dataset(
        {
            Region.from_state("MA"): {
                CommonFields.NEW_HOSPITAL_ADMISSIONS_COVID: [40, 50, 60, 50, 60, 40, 30, 40,],
                CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID: [40, 50, 60, 50, 60, 40,],
            },
        }
    )
    with pytest.raises(AssertionError):
        weekly_hospitalizations.add_weekly_hospitalizations(dataset_in=dataset_in)
