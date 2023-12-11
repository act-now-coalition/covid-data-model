import dataclasses
import io
from typing import List
import warnings

import pandas as pd
import pytest
from datapublic import common_df

from datapublic.common_fields import CommonFields
from datapublic.common_fields import FieldName
from freezegun import freeze_time

from libs.datasets.tail_filter import TagType
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral
from libs.datasets import timeseries
from libs.datasets.timeseries import DatasetName
from libs.metrics.test_positivity import Method
from libs.metrics.test_positivity import AllMethods
from libs.metrics.test_positivity import DivisionMethod
from libs.metrics import test_positivity
from libs.pipeline import Region


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error", "ignore::libs.pipeline.BadFipsWarning")


# NOTE (sean 2023-12-10): Ignore FutureWarnings due to pandas MultiIndex .loc deprecations.
@pytest.fixture(autouse=True)
def ignore_dependency_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)


def _parse_wide_dates(csv_str: str) -> pd.DataFrame:
    """Parses a string with columns for region, variable/provenance followed by dates."""
    # TODO(tom): At places where this is used try to replace DataFrame with MultiRegionDataset,
    #  then delete this function.
    df = pd.read_csv(io.StringIO(csv_str))
    df = df.set_index(list(df.columns[0:2]))
    df.columns = pd.to_datetime(df.columns)
    return df


def _replace_methods_attribute(methods: List[Method], **kwargs) -> List[Method]:
    """Returns a copy of the passed methods with attributes replaced according to kwargs."""
    return [dataclasses.replace(method, **kwargs) for method in methods]


def test_basic():
    region_as = Region.from_state("AS")
    region_tx = Region.from_state("TX")
    metrics_as = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([0, 2, 4, 6], provenance="pos"),
        CommonFields.TOTAL_TESTS: TimeseriesLiteral([100, 200, 300, 400]),
    }
    metrics_tx = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([1, 2, 3, 4], provenance="pos"),
        CommonFields.POSITIVE_TESTS_VIRAL: TimeseriesLiteral(
            [10, 20, 30, 40], provenance="pos_viral"
        ),
        CommonFields.TOTAL_TESTS: TimeseriesLiteral([100, 200, 300, 400]),
    }
    ds = test_helpers.build_dataset({region_as: metrics_as, region_tx: metrics_tx})

    methods = [
        DivisionMethod(
            DatasetName("method1"), CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method2"), CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS
        ),
    ]
    all_methods = AllMethods.run(ds, methods, diff_days=3)

    expected_positivity = test_helpers.build_dataset(
        {
            region_as: {CommonFields.TEST_POSITIVITY: TimeseriesLiteral([0.02], provenance="pos")},
            region_tx: {
                CommonFields.TEST_POSITIVITY: TimeseriesLiteral([0.1], provenance="pos_viral")
            },
        },
        start_date="2020-04-04",
    )
    test_helpers.assert_dataset_like(all_methods.test_positivity, expected_positivity)


def test_recent_days():
    region_as = Region.from_state("AS")
    region_tx = Region.from_state("TX")
    metrics_as = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([0, 2, 4, 6], provenance="pos"),
        CommonFields.POSITIVE_TESTS_VIRAL: TimeseriesLiteral(
            [0, 20, None, None], provenance="pos_viral"
        ),
        CommonFields.TOTAL_TESTS: TimeseriesLiteral([100, 200, 300, 400]),
    }
    metrics_tx = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([1, 2, 3, 4], provenance="pos"),
        CommonFields.POSITIVE_TESTS_VIRAL: TimeseriesLiteral(
            [10, 20, 30, 40], provenance="pos_viral"
        ),
        CommonFields.TOTAL_TESTS: TimeseriesLiteral([100, 200, 300, 400]),
    }
    ds = test_helpers.build_dataset({region_as: metrics_as, region_tx: metrics_tx})
    methods = [
        DivisionMethod(
            DatasetName("method1"), CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method2"), CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS
        ),
    ]
    methods = _replace_methods_attribute(methods, recent_days=2)
    all_methods = AllMethods.run(ds, methods, diff_days=1)

    expected_positivity = test_helpers.build_dataset(
        {
            region_as: {
                CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
                    [0.02, 0.02, 0.02], provenance="pos"
                )
            },
            region_tx: {
                CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
                    [0.1, 0.1, 0.1], provenance="pos_viral"
                )
            },
        },
        start_date="2020-04-02",
    )
    test_helpers.assert_dataset_like(all_methods.test_positivity, expected_positivity)
    assert all_methods.test_positivity.get_one_region(region_as).provenance == {
        CommonFields.TEST_POSITIVITY: ["pos"]
    }
    assert all_methods.test_positivity.get_one_region(region_tx).provenance == {
        CommonFields.TEST_POSITIVITY: ["pos_viral"]
    }

    methods = _replace_methods_attribute(methods, recent_days=3)
    all_methods = AllMethods.run(ds, methods, diff_days=1)
    positivity_provenance = all_methods.test_positivity.provenance
    assert positivity_provenance.loc["iso1:us#iso2:us-as"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "pos_viral"
    }
    assert positivity_provenance.loc["iso1:us#iso2:us-tx"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "pos_viral"
    }


def test_missing_column_for_one_method():
    ds = test_helpers.build_default_region_dataset(
        {
            CommonFields.POSITIVE_TESTS: [1, 2, 3, 4],
            CommonFields.POSITIVE_TESTS_VIRAL: TimeseriesLiteral(
                [10, 20, 30, 40], provenance="pos_viral"
            ),
            CommonFields.TOTAL_TESTS: [100, 200, 300, 400],
        }
    )
    methods = [
        DivisionMethod(
            DatasetName("method1"), CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method2"), CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method3"),
            CommonFields.POSITIVE_TESTS,
            CommonFields.TOTAL_TESTS_PEOPLE_VIRAL,
        ),
    ]
    methods = _replace_methods_attribute(methods, recent_days=4)
    assert (
        AllMethods.run(ds, methods, diff_days=1)
        .test_positivity.provenance.loc[test_helpers.DEFAULT_REGION.location_id]
        .at[CommonFields.TEST_POSITIVITY]
        == "pos_viral"
    )


def test_missing_columns_for_all_tests():
    ds = test_helpers.build_default_region_dataset(
        {FieldName("m1"): [1, 2, 3, 4], FieldName("m2"): [10, 20, 30, 40]}
    )
    methods = [
        DivisionMethod(
            DatasetName("method1"), CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method2"), CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method3"),
            CommonFields.POSITIVE_TESTS,
            CommonFields.TOTAL_TESTS_PEOPLE_VIRAL,
        ),
    ]
    methods = _replace_methods_attribute(methods, recent_days=4)
    with pytest.raises(test_positivity.NoMethodsWithRelevantColumns):
        AllMethods.run(ds, methods, diff_days=1)


def test_column_present_with_no_data():
    region_tx = Region.from_state("TX")
    ds = test_helpers.build_dataset(
        {region_tx: {CommonFields.TOTAL_TESTS: [100, 200, 400]}},
        timeseries_columns=[CommonFields.POSITIVE_TESTS],
    )
    method = DivisionMethod(
        DatasetName("method2"),
        CommonFields.POSITIVE_TESTS,
        CommonFields.TOTAL_TESTS,
        recent_days=1,
    )
    with pytest.raises(test_positivity.NoColumnsWithDataException):
        AllMethods.run(ds, [method], diff_days=1)


def test_all_columns_na():
    # MultiRegionDataset.from_csv drops columns with no real values so make a DataFrame
    # to pass to from_timeseries_df.
    ts_df = common_df.read_csv(
        io.StringIO(
            "location_id,date,positive_tests,total_tests\n"
            "iso1:us#iso2:us-tx,2020-04-01,,\n"
            "iso1:us#iso2:us-tx,2020-04-02,,\n"
            "iso1:us#iso2:us-tx,2020-04-04,,\n"
        ),
        set_index=False,
    )
    ts_df[CommonFields.POSITIVE_TESTS] = pd.NA
    ts = timeseries.MultiRegionDataset.from_timeseries_df(ts_df)
    methods = [
        DivisionMethod(
            DatasetName("method2"),
            CommonFields.POSITIVE_TESTS,
            CommonFields.TOTAL_TESTS,
            recent_days=1,
        ),
    ]
    with pytest.raises(test_positivity.NoRealTimeseriesValuesException):
        AllMethods.run(ts, methods, diff_days=1)


def test_provenance():
    region_as = Region.from_state("AS")
    region_tx = Region.from_state("TX")
    metrics_as = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([0, 2, 4, 6], provenance="pt_src1"),
        CommonFields.TOTAL_TESTS: [100, 200, 300, 400],
    }
    metrics_tx = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([1, 2, 3, 4], provenance="pt_src2"),
        CommonFields.POSITIVE_TESTS_VIRAL: TimeseriesLiteral(
            [10, 20, 30, 40], provenance="pos_viral"
        ),
        CommonFields.TOTAL_TESTS: [100, 200, 300, 400],
    }
    dataset_in = test_helpers.build_dataset({region_as: metrics_as, region_tx: metrics_tx})

    methods = [
        DivisionMethod(
            DatasetName("method1"), CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method2"), CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS
        ),
    ]
    all_methods = AllMethods.run(dataset_in, methods, diff_days=3)

    expected_as = {CommonFields.TEST_POSITIVITY: TimeseriesLiteral([0.02], provenance=["pt_src1"])}
    expected_tx = {CommonFields.TEST_POSITIVITY: TimeseriesLiteral([0.1], provenance="pos_viral")}
    expected_positivity = test_helpers.build_dataset(
        {region_as: expected_as, region_tx: expected_tx}, start_date="2020-04-04"
    )
    test_helpers.assert_dataset_like(all_methods.test_positivity, expected_positivity)


def test_preserve_tags():
    region_as = Region.from_state("AS")
    region_tx = Region.from_state("TX")
    tag1 = test_helpers.make_tag(TagType.CUMULATIVE_LONG_TAIL_TRUNCATED, date="2020-04-04")
    tag2 = test_helpers.make_tag(TagType.CUMULATIVE_TAIL_TRUNCATED, date="2020-04-04")
    tag_drop = test_helpers.make_tag(TagType.ZSCORE_OUTLIER, date="2020-04-01")
    tag3 = test_helpers.make_tag(TagType.ZSCORE_OUTLIER, date="2020-04-04")
    tag4 = test_helpers.make_tag(TagType.ZSCORE_OUTLIER, date="2020-04-03")
    metrics_as = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral(
            [1, 2, 3, 4], annotation=[tag1], provenance="pos"
        ),
        CommonFields.TOTAL_TESTS: TimeseriesLiteral([100, 200, 300, 400], annotation=[tag2]),
    }
    metrics_tx = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([None, None, 3, 4], annotation=[tag_drop]),
        CommonFields.POSITIVE_TESTS_VIRAL: [10, 20, 30, 40],
        CommonFields.TOTAL_TESTS: TimeseriesLiteral([100, 200, 300, 400], annotation=[tag3, tag4]),
    }
    dataset_in = test_helpers.build_dataset({region_as: metrics_as, region_tx: metrics_tx})

    methods = [
        DivisionMethod(
            DatasetName("method1"), CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS
        ),
        DivisionMethod(
            DatasetName("method2"), CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS
        ),
    ]
    all_methods = AllMethods.run(dataset_in, methods, diff_days=3)

    expected_as = {
        CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
            [0.01], provenance="pos", annotation=[tag1, tag2]
        )
    }
    expected_tx = {CommonFields.TEST_POSITIVITY: TimeseriesLiteral([0.1], annotation=[tag3, tag4])}
    expected_positivity = test_helpers.build_dataset(
        {region_as: expected_as, region_tx: expected_tx}, start_date="2020-04-04"
    )
    test_helpers.assert_dataset_like(all_methods.test_positivity, expected_positivity)


def test_default_positivity_methods():
    # This test intentionally doesn't pass any methods to AllMethods.run to run the methods used
    # in production.
    region_as = Region.from_state("AS")
    region_tx = Region.from_state("TX")
    metrics_as = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([0, 1, 2, 3, 4, 5, 6, 7], provenance="src1"),
        CommonFields.NEGATIVE_TESTS: TimeseriesLiteral(
            [10, 19, 28, 37, 46, 55, 64, 73], provenance="src1"
        ),
    }
    metrics_tx = {
        CommonFields.POSITIVE_TESTS_VIRAL: TimeseriesLiteral(
            [2, 4, 6, 8, 10, 12, 14, 16], provenance="pos_tests"
        ),
        CommonFields.TOTAL_TESTS_VIRAL: [10, 20, 30, 40, 50, 60, 70, 80],
    }
    dataset_in = test_helpers.build_dataset({region_as: metrics_as, region_tx: metrics_tx})

    # TODO(tom): Once test positivity code seems stable remove call to datetime.today() in
    #  has_recent_data and remove this freeze_time.
    with freeze_time("2020-04-14"):
        all_methods = AllMethods.run(dataset_in, diff_days=1)

    expected_as = {
        CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], provenance="src1",
        )
    }
    expected_tx = {
        CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], provenance="pos_tests"
        )
    }
    expected_positivity = test_helpers.build_dataset(
        {region_as: expected_as, region_tx: expected_tx}, start_date="2020-04-02",
    )
    test_helpers.assert_dataset_like(all_methods.test_positivity, expected_positivity)


@pytest.mark.parametrize("pos_neg_tests_recent", [False, True])
def test_recent_pos_neg_tests_has_positivity_ratio(pos_neg_tests_recent):
    # positive_tests and negative_tests appear on 8/10 and 8/11. They will be used when
    # that is within 10 days of 'today'.
    dataset_in = test_helpers.build_default_region_dataset(
        {
            CommonFields.TEST_POSITIVITY_7D: TimeseriesLiteral(
                [0.02, 0.03, 0.04, 0.05, 0.06, 0.07], provenance="CDCTesting"
            ),
            CommonFields.POSITIVE_TESTS: TimeseriesLiteral(
                [1, 2, None, None, None, None], provenance="pos"
            ),
            CommonFields.NEGATIVE_TESTS: [10, 20, None, None, None, None],
        },
        start_date="2020-08-10",
    )

    if pos_neg_tests_recent:
        freeze_date = "2020-08-21"
        # positive_tests and negative_tests are used
        expected_metrics = {
            CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
                [None, 2 / (2 + 20), None, None, None, None], provenance="pos"
            )
        }
        expected = test_helpers.build_default_region_dataset(
            expected_metrics, start_date="2020-08-10"
        )

    else:
        freeze_date = "2020-08-22"
        # positive_tests and negative_tests no longer recent so test_positivity_7d is copied to
        # output.
        expected_metrics = {
            CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
                [0.02, 0.03, 0.04, 0.05, 0.06, 0.07], provenance="CDCTesting"
            )
        }
        expected = test_helpers.build_default_region_dataset(
            expected_metrics, start_date="2020-08-10"
        )

    with freeze_time(freeze_date):
        all_methods = AllMethods.run(dataset_in)

    # check_less_precise so only 3 digits need match for testPositivityRatio
    test_helpers.assert_dataset_like(all_methods.test_positivity, expected)
