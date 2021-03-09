import pytest
from covidactnow.datapublic.common_fields import CommonFields
from tests import test_helpers
from libs.datasets import outlier_detection
from libs.datasets.taglib import TagType

TimeseriesLiteral = test_helpers.TimeseriesLiteral


def test_remove_outliers():
    values = [10.0] * 7 + [1000.0]
    dataset = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values})
    dataset = outlier_detection.drop_new_case_outliers(dataset)

    # Expected result is the same series with the last value removed
    expected_tag = test_helpers.make_tag(
        TagType.ZSCORE_OUTLIER, date="2020-04-08", original_observation=1000.0,
    )
    expected_ts = TimeseriesLiteral([10.0] * 7, annotation=[expected_tag])
    expected = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: expected_ts})
    test_helpers.assert_dataset_like(dataset, expected, drop_na_dates=True)


def test_remove_outliers_threshold():
    values = [1.0] * 7 + [30.0]
    dataset = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values})
    result = outlier_detection.drop_series_outliers(dataset, CommonFields.NEW_CASES, threshold=30)

    # Should not modify becasue not higher than threshold
    test_helpers.assert_dataset_like(dataset, result)

    result = outlier_detection.drop_series_outliers(dataset, CommonFields.NEW_CASES, threshold=29)

    # Expected result is the same series with the last value removed
    expected_tag = test_helpers.make_tag(
        TagType.ZSCORE_OUTLIER, date="2020-04-08", original_observation=30.0
    )
    expected_ts = TimeseriesLiteral([1.0] * 7, annotation=[expected_tag])
    expected = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: expected_ts})
    test_helpers.assert_dataset_like(result, expected, drop_na_dates=True)


def test_not_removing_short_series():
    values = [None] * 7 + [1, 1, 300]
    dataset = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values})
    result = outlier_detection.drop_series_outliers(dataset, CommonFields.NEW_CASES, threshold=30)

    test_helpers.assert_dataset_like(dataset, result)


# TODO(chris): Make test stronger, doesn't cover all edge cases
@pytest.mark.parametrize("last_value,is_outlier", [(0.02, False), (0.045, True)])
def test_remove_test_positivity_outliers(last_value, is_outlier):
    values = [0.015] * 7 + [last_value]
    dataset_in = test_helpers.build_default_region_dataset(
        {CommonFields.TEST_POSITIVITY_7D: values}
    )
    dataset_out = outlier_detection.drop_tail_positivity_outliers(dataset_in)

    # Expected result is the same series with the last value removed
    if is_outlier:
        expected_tag = test_helpers.make_tag(
            TagType.ZSCORE_OUTLIER, date="2020-04-08", original_observation=last_value,
        )
        expected_ts = TimeseriesLiteral([0.015] * 7, annotation=[expected_tag])
        expected = test_helpers.build_default_region_dataset(
            {CommonFields.TEST_POSITIVITY_7D: expected_ts}
        )
        test_helpers.assert_dataset_like(dataset_out, expected, drop_na_dates=True)

    else:
        test_helpers.assert_dataset_like(dataset_in, dataset_out, drop_na_dates=True)
