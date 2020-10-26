import io
import pandas as pd
import pytest
from covidactnow.datapublic import common_df

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import timeseries
from libs.test_positivity import AllMethods
from libs.test_positivity import Method
from libs import test_positivity
from test.libs.datasets.timeseries_test import assert_combined_like


def _parse_wide_dates(csv_str: str) -> pd.DataFrame:
    """Parses a string with columns for region, variable/provenance followed by dates."""
    df = pd.read_csv(io.StringIO(csv_str))
    df = df.set_index(list(df.columns[0:2]))
    df.columns = pd.to_datetime(df.columns)
    return df


def test_basic():
    ts = timeseries.MultiRegionTimeseriesDataset.from_csv(
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
        Method("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        Method("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
    ]
    all_methods = AllMethods.run(ts, methods, 3, 14)

    expected_df = _parse_wide_dates(
        "location_id,variable,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "iso1:us#iso2:as,method2,,,,0.02\n"
        "iso1:us#iso2:tx,method1,,,,0.1\n"
        "iso1:us#iso2:tx,method2,,,,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_df, check_like=True)

    expected_positivity = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,test_positivity\n"
            "iso1:us#iso2:as,2020-04-04,0.02\n"
            "iso1:us#iso2:tx,2020-04-04,0.1\n"
        )
    )
    assert_combined_like(all_methods.test_positivity, expected_positivity)
    positivity_provenance = all_methods.test_positivity.provenance
    # Use loc[...].at[...] as work-around for https://github.com/pandas-dev/pandas/issues/26989
    assert positivity_provenance.loc["iso1:us#iso2:as"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method2"
    }
    assert positivity_provenance.loc["iso1:us#iso2:tx"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method1"
    }


def test_recent_days():
    ts = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,positive_tests,positive_tests_viral,total_tests,\n"
            "iso1:us#iso2:as,2020-04-01,0,0,100\n"
            "iso1:us#iso2:as,2020-04-02,2,20,200\n"
            "iso1:us#iso2:as,2020-04-03,4,,300\n"
            "iso1:us#iso2:as,2020-04-04,6,,400\n"
            "iso1:us#iso2:tx,2020-04-01,1,10,100\n"
            "iso1:us#iso2:tx,2020-04-02,2,20,200\n"
            "iso1:us#iso2:tx,2020-04-03,3,30,300\n"
            "iso1:us#iso2:tx,2020-04-04,4,40,400\n"
        )
    )
    methods = [
        Method("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        Method("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
    ]
    all_methods = AllMethods.run(ts, methods, diff_days=1, recent_days=2)

    expected_all = _parse_wide_dates(
        "location_id,variable,2020-04-01,2020-04-02,2020-04-03,2020-04-04\n"
        "iso1:us#iso2:as,method1,,0.2,,\n"
        "iso1:us#iso2:as,method2,,0.02,0.02,0.02\n"
        "iso1:us#iso2:tx,method1,,0.1,0.1,0.1\n"
        "iso1:us#iso2:tx,method2,,0.01,0.01,0.01\n"
    )
    pd.testing.assert_frame_equal(all_methods.all_methods_timeseries, expected_all, check_like=True)
    expected_positivity = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,test_positivity\n"
            "iso1:us#iso2:as,2020-04-02,0.02\n"
            "iso1:us#iso2:as,2020-04-03,0.02\n"
            "iso1:us#iso2:as,2020-04-04,0.02\n"
            "iso1:us#iso2:tx,2020-04-02,0.1\n"
            "iso1:us#iso2:tx,2020-04-03,0.1\n"
            "iso1:us#iso2:tx,2020-04-04,0.1\n"
        )
    )
    assert_combined_like(all_methods.test_positivity, expected_positivity)
    # Use loc[...].at[...] as work-around for https://github.com/pandas-dev/pandas/issues/26989
    positivity_provenance = all_methods.test_positivity.provenance
    assert positivity_provenance.loc["iso1:us#iso2:as"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method2"
    }
    assert positivity_provenance.loc["iso1:us#iso2:tx"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method1"
    }

    all_methods = AllMethods.run(ts, methods, diff_days=1, recent_days=4)
    positivity_provenance = all_methods.test_positivity.provenance
    assert positivity_provenance.loc["iso1:us#iso2:as"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method1"
    }
    assert positivity_provenance.loc["iso1:us#iso2:tx"].to_dict() == {
        CommonFields.TEST_POSITIVITY: "method1"
    }


def test_missing_column_for_one_method():
    ts = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,positive_tests,positive_tests_viral,total_tests\n"
            "iso1:us#iso2:tx,2020-04-01,1,10,100\n"
            "iso1:us#iso2:tx,2020-04-02,2,20,200\n"
            "iso1:us#iso2:tx,2020-04-03,3,30,300\n"
            "iso1:us#iso2:tx,2020-04-04,4,40,400\n"
        )
    )
    methods = [
        Method("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        Method("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
        Method("method3", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS_PEOPLE_VIRAL),
    ]
    assert (
        AllMethods.run(ts, methods, diff_days=1, recent_days=4)
        .test_positivity.provenance.loc["iso1:us#iso2:tx"]
        .at[CommonFields.TEST_POSITIVITY]
        == "method1"
    )


def test_missing_columns_for_all_tests():
    ts = timeseries.MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "location_id,date,m1,m2,m3\n"
            "iso1:us#iso2:tx,2020-04-01,1,10,100\n"
            "iso1:us#iso2:tx,2020-04-02,2,20,200\n"
            "iso1:us#iso2:tx,2020-04-03,3,30,300\n"
            "iso1:us#iso2:tx,2020-04-04,4,40,400\n"
        )
    )
    methods = [
        Method("method1", CommonFields.POSITIVE_TESTS_VIRAL, CommonFields.TOTAL_TESTS),
        Method("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
        Method("method3", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS_PEOPLE_VIRAL),
    ]
    with pytest.raises(test_positivity.NoMethodsWithRelevantColumns):
        AllMethods.run(ts, methods, diff_days=1, recent_days=4)


def test_column_present_with_no_data():
    # MultiRegionTimeseriesDataset.from_csv drops columns with no real values so make a DataFrame
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
    ts = timeseries.MultiRegionTimeseriesDataset.from_timeseries_df(ts_df)
    methods = [
        Method("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
    ]
    with pytest.raises(test_positivity.NoColumnsWithDataException):
        AllMethods.run(ts, methods, diff_days=1, recent_days=1)


def test_all_columns_na():
    # MultiRegionTimeseriesDataset.from_csv drops columns with no real values so make a DataFrame
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
    ts = timeseries.MultiRegionTimeseriesDataset.from_timeseries_df(ts_df)
    methods = [
        Method("method2", CommonFields.POSITIVE_TESTS, CommonFields.TOTAL_TESTS),
    ]
    with pytest.raises(test_positivity.NoRealTimeseriesValuesException):
        AllMethods.run(ts, methods, diff_days=1, recent_days=1)
