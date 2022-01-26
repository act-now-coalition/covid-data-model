import pytest

from datapublic.common_fields import CommonFields

from libs.datasets.taglib import TagType
from libs.datasets.tail_filter import TailFilter

from tests import test_helpers


def _assert_tail_filter_counts(
    tail_filter: TailFilter,
    *,
    skipped_too_short: int = 0,
    skipped_na_mean: int = 0,
    all_good: int = 0,
    truncated: int = 0,
    long_truncated: int = 0,
):
    """Asserts that tail_filter has given attribute count values, defaulting to zero."""
    assert tail_filter.skipped_too_short == skipped_too_short
    assert tail_filter.skipped_na_mean == skipped_na_mean
    assert tail_filter.all_good == all_good
    assert tail_filter.truncated == truncated
    assert tail_filter.long_truncated == long_truncated


def test_tail_filter_stalled_timeseries():
    # Make a timeseries that has 24 days increasing.
    values_increasing = list(range(100_000, 124_000, 1_000))
    # Add 4 days that copy the 24th day. The filter is meant to remove these.
    values_stalled = values_increasing + [values_increasing[-1]] * 4
    assert len(values_stalled) == 28

    ds_in = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values_stalled})
    tail_filter, ds_out = TailFilter.run(ds_in, [CommonFields.NEW_CASES])
    _assert_tail_filter_counts(tail_filter, truncated=1)
    tag_content = (
        "Removed 4 observations that look suspicious compared to mean diff of 1000.0 a few weeks "
        "ago."
    )
    truncated_timeseries = test_helpers.TimeseriesLiteral(
        values_increasing,
        annotation=[
            test_helpers.make_tag(
                TagType.CUMULATIVE_TAIL_TRUNCATED, date="2020-04-24", original_observation=123_000.0
            )
        ],
    )
    ds_expected = test_helpers.build_default_region_dataset(
        {CommonFields.NEW_CASES: truncated_timeseries}
    )
    test_helpers.assert_dataset_like(ds_out, ds_expected)

    # Try again with one day less, not enough for the filter so it returns the data unmodified.
    ds_in = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values_stalled[:-1]})
    tail_filter, ds_out = TailFilter.run(ds_in, [CommonFields.NEW_CASES])
    _assert_tail_filter_counts(tail_filter, skipped_too_short=1)
    test_helpers.assert_dataset_like(ds_out, ds_in)


def test_tail_filter_mean_nan():
    # Make a timeseries that has 14 days of NaN, than 14 days of increasing values. The first
    # 100_000 is there so the NaN form a gap that isn't dropped by unrelated code.
    values = [100_000] + [float("NaN")] * 14 + list(range(100_000, 114_000, 1_000))
    assert len(values) == 29

    ds_in = test_helpers.build_default_region_dataset({CommonFields.NEW_CASES: values})
    tail_filter, ds_out = TailFilter.run(ds_in, [CommonFields.NEW_CASES])
    _assert_tail_filter_counts(tail_filter, skipped_na_mean=1)
    test_helpers.assert_dataset_like(ds_out, ds_in, drop_na_dates=True)


def test_tail_filter_two_series():
    # Check that two series are both filtered. Currently the 'good' dates of 14-28 days ago are
    # relative to the most recent date of any timeseries but maybe it should be per-timeseries.
    pos_tests = list(range(100_000, 128_000, 1_000))
    tot_tests = list(range(1_000_000, 1_280_000, 10_000))
    # Pad positive tests with two 'None's so the timeseries are the same length.
    pos_tests_stalled = pos_tests + [pos_tests[-1]] * 3 + [None] * 2
    tot_tests_stalled = tot_tests + [tot_tests[-1]] * 5

    ds_in = test_helpers.build_default_region_dataset(
        {
            CommonFields.POSITIVE_TESTS: pos_tests_stalled,
            CommonFields.TOTAL_TESTS: tot_tests_stalled,
        }
    )
    tail_filter, ds_out = TailFilter.run(
        ds_in, [CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS]
    )
    ds_expected = test_helpers.build_default_region_dataset(
        {CommonFields.POSITIVE_TESTS: pos_tests, CommonFields.TOTAL_TESTS: tot_tests}
    )
    _assert_tail_filter_counts(tail_filter, truncated=2)
    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True, compare_tags=False)


def test_tail_filter_diff_goes_negative():
    # The end of this timeseries is (in 1000s) ... 127, 126, 127, 127. Ony the last 127 is
    # expected to be truncated.
    values = list(range(100_000, 128_000, 1_000)) + [126_000, 127_000, 127_000]
    assert len(values) == 31

    ds_in = test_helpers.build_default_region_dataset({CommonFields.CASES: values})
    tail_filter, ds_out = TailFilter.run(ds_in, [CommonFields.CASES])
    ds_expected = test_helpers.build_default_region_dataset({CommonFields.CASES: values[:-1]})
    _assert_tail_filter_counts(tail_filter, truncated=1)
    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True, compare_tags=False)


def test_tail_filter_zero_diff():
    # Make sure constant value timeseries is not truncated.
    values = [100_000] * 28

    ds_in = test_helpers.build_default_region_dataset({CommonFields.CASES: values})
    tail_filter, ds_out = TailFilter.run(ds_in, [CommonFields.CASES])
    _assert_tail_filter_counts(tail_filter, all_good=1)
    test_helpers.assert_dataset_like(ds_out, ds_in, drop_na_dates=True)


@pytest.mark.parametrize("stall_count", [0, 1, 2, 4])
def test_tail_filter_small_diff(stall_count: int):
    # Make sure a zero increase in the most recent value(s) of a series that was increasing
    # slowly is not dropped.
    values = list(range(1_000, 1_030)) + [1_029] * stall_count

    ds_in = test_helpers.build_default_region_dataset({CommonFields.CASES: values})
    tail_filter, ds_out = TailFilter.run(ds_in, [CommonFields.CASES])
    _assert_tail_filter_counts(tail_filter, all_good=1)
    test_helpers.assert_dataset_like(ds_out, ds_in, drop_na_dates=True)


@pytest.mark.parametrize(
    "stall_count,annotation_type",
    [
        (6, TagType.CUMULATIVE_TAIL_TRUNCATED),
        (7, TagType.CUMULATIVE_TAIL_TRUNCATED),
        (8, TagType.CUMULATIVE_LONG_TAIL_TRUNCATED),
        (9, TagType.CUMULATIVE_LONG_TAIL_TRUNCATED),
        (13, TagType.CUMULATIVE_LONG_TAIL_TRUNCATED),
        (14, TagType.CUMULATIVE_LONG_TAIL_TRUNCATED),
        (15, TagType.CUMULATIVE_LONG_TAIL_TRUNCATED),
        (16, TagType.CUMULATIVE_LONG_TAIL_TRUNCATED),
    ],
)
def test_tail_filter_long_stall(stall_count: int, annotation_type: TagType):
    # This timeseries has stalled for a long time.
    values = list(range(100_000, 128_000, 1_000)) + [127_000] * stall_count
    assert len(values) == 28 + stall_count

    ds_in = test_helpers.build_default_region_dataset({CommonFields.CASES: values})
    tail_filter, ds_out = TailFilter.run(ds_in, [CommonFields.CASES])
    # There are never more than 13 stalled observations removed.
    ds_expected = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: values[: -min(stall_count, 14)]}
    )
    if annotation_type is TagType.CUMULATIVE_TAIL_TRUNCATED:
        _assert_tail_filter_counts(tail_filter, truncated=1)
    elif annotation_type is TagType.CUMULATIVE_LONG_TAIL_TRUNCATED:
        _assert_tail_filter_counts(tail_filter, long_truncated=1)

    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True, compare_tags=False)


# TODO(tom): Add test with bucket not "all"
