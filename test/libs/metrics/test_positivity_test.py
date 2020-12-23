import dataclasses
import io
from typing import List
from typing import Union

import pandas as pd
import pytest
from covidactnow.datapublic import common_df

from covidactnow.datapublic.common_fields import CommonFields
from freezegun import freeze_time

from libs.datasets import timeseries
from libs.pipeline import Region
from libs.metrics.test_positivity import AllMethods
from libs.metrics.test_positivity import DivisionMethod
from libs.metrics.test_positivity import PassThruMethod
from libs.metrics import test_positivity
from test.libs.datasets.timeseries_test import assert_dataset_like
from test.libs.metrics import top_level_metrics_test
from test.libs.metrics.top_level_metrics_test import TimeseriesLiteral


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error", "ignore::libs.pipeline.BadFipsWarning")


def _parse_wide_dates(csv_str: str) -> pd.DataFrame:
    """Parses a string with columns for region, variable/provenance followed by dates."""
    # TODO(tom): At places where this is used try to replace DataFrame with MultiRegionDataset,
    #  then delete this function.
    df = pd.read_csv(io.StringIO(csv_str))
    df = df.set_index(list(df.columns[0:2]))
    df.columns = pd.to_datetime(df.columns)
    return df


def _replace_methods_attribute(
    methods: List[Union[DivisionMethod, PassThruMethod]], **kwargs
) -> List[Union[DivisionMethod, PassThruMethod]]:
    return [dataclasses.replace(method, **kwargs) for method in methods]


def test_basic():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,positive_tests,positive_tests_viral,total_tests,\n"
            "iso1:us#iso2:as,2020-04-01,0,,100\n"
            "iso1:us#iso2:as,2020-04-02,2,,200\n"
            "iso1:us#iso2:as,2020-04-03,4,,300\n"
            "iso1:us#iso2:as,2020-04-04,6,,400\n"
            "iso1:us#iso2:tx,2020-04-01,1,10,100\n"
            "iso1:us#iso2:tx,2020-04-02,2,20,200\n"
            "iso1:us#iso2:tx,2020-04-03,3,30,300\n"
            "iso1:us#iso2:tx,2020-04-04,4,40,400\n"
        )
    )

    methods = [
        DivisionMethod("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        DivisionMethod("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
    ]
    all_methods = AllMethods.run(ts, methods, 3)

    expected_df = _parse_wide_dates(
        "location_id,dataset,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "iso1:us#iso2:as,method2,,,,0.02\n"
        "iso1:us#iso2:tx,method1,,,,0.1\n"
        "iso1:us#iso2:tx,method2,,,,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_df, check_like=True)

    expected_positivity = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,test_positivity\n"
            "iso1:us#iso2:as,2020-04-04,0.02\n"
            "iso1:us#iso2:tx,2020-04-04,0.1\n"
        )
    ).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#iso2:as,test_positivity,method2\n"
            "iso1:us#iso2:tx,test_positivity,method1\n"
        )
    )
    assert_dataset_like(all_methods.test_positivity, expected_positivity)


def test_recent_days():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,positive_tests,positive_tests_viral,total_tests,\n"
            "iso1:us#iso2:us-as,2020-04-01,0,0,100\n"
            "iso1:us#iso2:us-as,2020-04-02,2,20,200\n"
            "iso1:us#iso2:us-as,2020-04-03,4,,300\n"
            "iso1:us#iso2:us-as,2020-04-04,6,,400\n"
            "iso1:us#iso2:us-tx,2020-04-01,1,10,100\n"
            "iso1:us#iso2:us-tx,2020-04-02,2,20,200\n"
            "iso1:us#iso2:us-tx,2020-04-03,3,30,300\n"
            "iso1:us#iso2:us-tx,2020-04-04,4,40,400\n"
        )
    )
    methods = [
        DivisionMethod("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        DivisionMethod("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
    ]
    methods = _replace_methods_attribute(methods, recent_days=2)
    all_methods = AllMethods.run(ts, methods, diff_days=1)

    expected_all = _parse_wide_dates(
        "location_id,dataset,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "iso1:us#iso2:us-as,method1,,0.2,,\n"
        "iso1:us#iso2:us-as,method2,,0.02,0.02,0.02\n"
        "iso1:us#iso2:us-tx,method1,,0.1,0.1,0.1\n"
        "iso1:us#iso2:us-tx,method2,,0.01,0.01,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_all, check_like=True)
    expected_positivity = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,test_positivity\n"
            "iso1:us#iso2:us-as,2020-04-02,0.02\n"
            "iso1:us#iso2:us-as,2020-04-03,0.02\n"
            "iso1:us#iso2:us-as,2020-04-04,0.02\n"
            "iso1:us#iso2:us-tx,2020-04-02,0.1\n"
            "iso1:us#iso2:us-tx,2020-04-03,0.1\n"
            "iso1:us#iso2:us-tx,2020-04-04,0.1\n"
        )
    ).add_provenance_csv(
        io.StringIO(
            "location_id,variable,provenance\n"
            "iso1:us#iso2:us-as,test_positivity,method2\n"
            "iso1:us#iso2:us-tx,test_positivity,method1\n"
        )
    )
    assert_dataset_like(all_methods.test_positivity, expected_positivity)
    assert all_methods.test_positivity.get_one_region(Region.from_state("AS")).provenance == {
        CommonFields.TEST_POSITIVITY: "method2"
    }
    assert all_methods.test_positivity.get_one_region(Region.from_state("TX")).provenance == {
        CommonFields.TEST_POSITIVITY: "method1"
    }

    methods = _replace_methods_attribute(methods, recent_days=3)
    all_methods = AllMethods.run(ts, methods, diff_days=1)
    positivity_provenance = all_methods.test_positivity.provenance
    assert positivity_provenance.loc["iso1:us#iso2:us-as"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method1"
    }
    assert positivity_provenance.loc["iso1:us#iso2:us-tx"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method1"
    }


def test_missing_column_for_one_method():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,positive_tests,positive_tests_viral,total_tests\n"
            "iso1:us#iso2:tx,2020-04-01,1,10,100\n"
            "iso1:us#iso2:tx,2020-04-02,2,20,200\n"
            "iso1:us#iso2:tx,2020-04-03,3,30,300\n"
            "iso1:us#iso2:tx,2020-04-04,4,40,400\n"
        )
    )
    methods = [
        DivisionMethod("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        DivisionMethod("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
        DivisionMethod(
            "method3", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS_PEOPLE_VIRAL
        ),
    ]
    methods = _replace_methods_attribute(methods, recent_days=4)
    assert (
        AllMethods.run(ts, methods, diff_days=1)
        .test_positivity.provenance.loc["iso1:us#iso2:tx"]
        .at[CommonFields.TEST_POSITIVITY]
        == "method1"
    )


def test_missing_columns_for_all_tests():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,m1,m2,m3\n"
            "iso1:us#iso2:tx,2020-04-01,1,10,100\n"
            "iso1:us#iso2:tx,2020-04-02,2,20,200\n"
            "iso1:us#iso2:tx,2020-04-03,3,30,300\n"
            "iso1:us#iso2:tx,2020-04-04,4,40,400\n"
        )
    )
    methods = [
        DivisionMethod("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        DivisionMethod("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
        DivisionMethod(
            "method3", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS_PEOPLE_VIRAL
        ),
    ]
    methods = _replace_methods_attribute(methods, recent_days=4)
    with pytest.raises(test_positivity.NoMethodsWithRelevantColumns):
        AllMethods.run(ts, methods, diff_days=1)


def test_column_present_with_no_data():
    # MultiRegionDataset.from_csv drops columns with no real values so make a DataFrame
    # to pass to from_timeseries_df.
    ts_df = common_df.read_csv(
        io.StringIO(
            "location_id,date,positive_tests,total_tests\n"
            "iso1:us#iso2:tx,2020-04-01,,100\n"
            "iso1:us#iso2:tx,2020-04-02,,200\n"
            "iso1:us#iso2:tx,2020-04-04,,400\n"
        ),
        set_index=False,
    )
    ts_df[CommonFields.POSITIVE_TESTS] = pd.NA
    ts = timeseries.MultiRegionDataset.from_geodata_timeseries_df(ts_df)
    methods = [
        DivisionMethod(
            "method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS, recent_days=1
        ),
    ]
    with pytest.raises(test_positivity.NoColumnsWithDataException):
        AllMethods.run(ts, methods, diff_days=1)


def test_all_columns_na():
    # MultiRegionDataset.from_csv drops columns with no real values so make a DataFrame
    # to pass to from_timeseries_df.
    ts_df = common_df.read_csv(
        io.StringIO(
            "location_id,date,positive_tests,total_tests\n"
            "iso1:us#iso2:tx,2020-04-01,,\n"
            "iso1:us#iso2:tx,2020-04-02,,\n"
            "iso1:us#iso2:tx,2020-04-04,,\n"
        ),
        set_index=False,
    )
    ts_df[CommonFields.POSITIVE_TESTS] = pd.NA
    ts = timeseries.MultiRegionDataset.from_geodata_timeseries_df(ts_df)
    methods = [
        DivisionMethod(
            "method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS, recent_days=1
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
        CommonFields.POSITIVE_TESTS_VIRAL: [10, 20, 30, 40],
        CommonFields.TOTAL_TESTS: [100, 200, 300, 400],
    }
    dataset_in = top_level_metrics_test.build_dataset(
        {region_as: metrics_as, region_tx: metrics_tx}
    )

    methods = [
        DivisionMethod("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        DivisionMethod("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
    ]
    all_methods = AllMethods.run(dataset_in, methods, 3)

    expected_df = _parse_wide_dates(
        "location_id,dataset,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "iso1:us#iso2:us-as,method2,,,,0.02\n"
        "iso1:us#iso2:us-tx,method1,,,,0.1\n"
        "iso1:us#iso2:us-tx,method2,,,,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_df, check_like=True)

    expected_as = {CommonFields.TEST_POSITIVITY: TimeseriesLiteral([0.02], provenance="method2")}
    expected_tx = {CommonFields.TEST_POSITIVITY: TimeseriesLiteral([0.1], provenance="method1")}
    expected_positivity = top_level_metrics_test.build_dataset(
        {region_as: expected_as, region_tx: expected_tx}, start_date="2020-04-04",
    )
    assert_dataset_like(all_methods.test_positivity, expected_positivity)


def test_default_methods():
    region_as = Region.from_state("AS")
    region_tx = Region.from_state("TX")
    metrics_as = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([0, 1, 2, 3, 4, 5, 6, 7], provenance="src1"),
        CommonFields.NEGATIVE_TESTS: TimeseriesLiteral(
            [10, 19, 28, 37, 46, 55, 64, 73], provenance="src2"
        ),
    }
    metrics_tx = {
        CommonFields.POSITIVE_TESTS: TimeseriesLiteral([1, 2, 3, 4], provenance="pt_src2"),
        CommonFields.POSITIVE_TESTS_VIRAL: [10, 20, 30, 40],
        CommonFields.TOTAL_TESTS: [100, 200, 300, 400],
    }
    dataset_in = top_level_metrics_test.build_dataset({region_as: metrics_as})

    # TODO(tom): Once test positivity code seems stable remove call to datetime.today() in
    #  has_recent_data and remove this freeze_time.
    with freeze_time("2020-04-14"):
        all_methods = AllMethods.run(dataset_in, diff_days=1)

    expected_as = {
        CommonFields.TEST_POSITIVITY: TimeseriesLiteral(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], provenance="OldMethod(src1," "src2)"
        )
    }
    expected_positivity = top_level_metrics_test.build_dataset(
        {region_as: expected_as}, start_date="2020-04-02",
    )
    assert_dataset_like(all_methods.test_positivity, expected_positivity)

    expected_df = _parse_wide_dates(
        "location_id,dataset,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "iso1:us#iso2:us-as,method2,,,,0.02\n"
        "iso1:us#iso2:us-tx,method1,,,,0.1\n"
        "iso1:us#iso2:us-tx,method2,,,,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_df, check_like=True)
