import pytest
import pandas as pd
from datapublic.common_fields import CommonFields
from datapublic.common_fields import DemographicBucket

from tests import test_helpers
from libs.datasets import outlier_detection
from libs.datasets.taglib import TagType
import datetime

TimeseriesLiteral = test_helpers.TimeseriesLiteral


@pytest.mark.parametrize(
    "start_date,final_value,outlier_expected", [("2020-06-02", 40, True), ("2021-06-02", 40, False)]
)
def test_remove_outliers_secondary_threshold(start_date, final_value, outlier_expected):
    # This test relies on the `secondary_zscore_threshold_after_date` setting
    # from `drop_new_case_outliers`.
    # The two inputs test the same value, making sure that one (before the new zscore
    # threshold is set) is not an outlier, while that same value after the
    # secondary threshold is set is an outlier.
    values = [5.0] * 3 + [10.0] * 3 + [final_value]
    dataset = test_helpers.build_default_region_dataset(
        {CommonFields.NEW_CASES: values}, start_date=start_date
    )
    dataset = outlier_detection.drop_new_case_outliers(dataset)

    new_cases = dataset.timeseries[CommonFields.NEW_CASES]

    if outlier_expected:
        # Double check that the last value (the outlier value) generates a tag.
        assert len(dataset.tag_objects_series) == 1
        pd.isna(new_cases.values[-1])
    else:
        # Double check that no tags are generated and final value is still in the
        # dataset.
        assert len(dataset.tag_objects_series) == 0
        assert new_cases.values[-1] == final_value


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


def test_remove_outliers_excludes_july_5th():
    test_start_date = datetime.datetime.strptime(test_helpers.DEFAULT_START_DATE, "%Y-%m-%d").date()
    normal_days = (datetime.date(2021, 7, 5) - test_start_date).days
    values = [1.0] * normal_days + [31.0]
    dataset = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values})
    result = outlier_detection.drop_new_case_outliers(dataset)

    # Should not remove outlier because 2021/07/05 is in the range of dates
    # excluded for outlier detection.
    test_helpers.assert_dataset_like(dataset, result)

    # Now test 2021/07/04.
    normal_days = normal_days - 1
    values = [1.0] * normal_days + [31.0]
    dataset = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values})
    result = outlier_detection.drop_new_case_outliers(dataset)

    # Should remove outlier because 2021/07/04 is not in the range.
    expected_tag = test_helpers.make_tag(
        TagType.ZSCORE_OUTLIER, date="2021-07-04", original_observation=31.0
    )
    expected_ts = TimeseriesLiteral([1.0] * normal_days, annotation=[expected_tag])
    expected = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: expected_ts})
    test_helpers.assert_dataset_like(result, expected, drop_na_dates=True)


def test_not_removing_short_series():
    values = [None] * 7 + [1, 1, 300]
    dataset = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values})
    result = outlier_detection.drop_series_outliers(dataset, CommonFields.NEW_CASES, threshold=30)

    test_helpers.assert_dataset_like(dataset, result)


def test_drop_series_outliers_preserves_buckets():
    age_40s = DemographicBucket("age:40-49")
    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.NEW_CASES: [1, 2, 3], CommonFields.CASES: {age_40s: [0, 1, 2]}}
    )
    ds_out = outlier_detection.drop_series_outliers(ds_in, CommonFields.NEW_CASES, threshold=30)

    test_helpers.assert_dataset_like(ds_in, ds_out)


def test_drop_series_outliers_remove_from_bucketed():
    age_40s = DemographicBucket("age:40-49")
    ts_unmodified = {DemographicBucket.ALL: [2.0] * 8}
    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.NEW_CASES: {age_40s: [1.0] * 7 + [32.0], **ts_unmodified}}
    )

    ds_out = outlier_detection.drop_series_outliers(ds_in, CommonFields.NEW_CASES, threshold=30)

    expected_tag = test_helpers.make_tag(
        TagType.ZSCORE_OUTLIER, date="2020-04-08", original_observation=32.0
    )
    ts_with_outlier_removed = TimeseriesLiteral([1.0] * 7 + [None], annotation=[expected_tag])
    ds_expected = test_helpers.build_default_region_dataset(
        {CommonFields.NEW_CASES: {age_40s: ts_with_outlier_removed, **ts_unmodified}}
    )
    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True)


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
