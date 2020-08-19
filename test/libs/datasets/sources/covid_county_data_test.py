import datetime
import pathlib
import pandas as pd
import pytest

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_test_helpers import to_dict
from libs.datasets import can_model_output_schema as schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.sources.covid_county_data import CovidCountyDataDataSource
from libs.enums import Intervention


# turns all warnings into errors for this module
from test.dataset_utils_test import read_csv_and_index_fips_date

pytestmark = pytest.mark.filterwarnings("error")


def test_fix_tests_and_cases():
    df = read_csv_and_index_fips_date(
        "fips,date,negative_tests,positive_tests,total_tests,cases\n"
        "97123,2020-04-01,9,1,10,1\n"
        "97123,2020-04-02,,,20,2\n"
    ).reset_index()

    result_df, provenance = CovidCountyDataDataSource.synthesize_test_metrics(df)

    assert to_dict([CommonFields.FIPS, CommonFields.DATE], result_df) == {
        ("97123", pd.to_datetime("2020-04-01")): {
            CommonFields.NEGATIVE_TESTS: 9,
            CommonFields.POSITIVE_TESTS: 1,
            CommonFields.TOTAL_TESTS: 10,
            CommonFields.CASES: 1,
        },
        ("97123", pd.to_datetime("2020-04-02")): {
            CommonFields.NEGATIVE_TESTS: 18,
            CommonFields.POSITIVE_TESTS: 2,
            CommonFields.TOTAL_TESTS: 20,
            CommonFields.CASES: 2,
        },
    }
    assert provenance.to_dict() == {
        ("97123", CommonFields.NEGATIVE_TESTS): "none;tests_and_cases",
        ("97123", CommonFields.POSITIVE_TESTS): "none;tests_and_cases",
    }


def test_fix_missing_neg():
    df = read_csv_and_index_fips_date(
        "fips,date,negative_tests,positive_tests,total_tests,cases\n"
        "97123,2020-04-01,9,1,10,1\n"
        "97123,2020-04-02,,3,20,2\n"
    ).reset_index()

    result_df, provenance = CovidCountyDataDataSource.synthesize_test_metrics(df)

    assert to_dict([CommonFields.FIPS, CommonFields.DATE], result_df) == {
        ("97123", pd.to_datetime("2020-04-01")): {
            CommonFields.NEGATIVE_TESTS: 9,
            CommonFields.POSITIVE_TESTS: 1,
            CommonFields.TOTAL_TESTS: 10,
            CommonFields.CASES: 1,
        },
        ("97123", pd.to_datetime("2020-04-02")): {
            CommonFields.NEGATIVE_TESTS: 17,
            CommonFields.POSITIVE_TESTS: 3,
            CommonFields.TOTAL_TESTS: 20,
            CommonFields.CASES: 2,
        },
    }
    assert provenance.to_dict() == {
        ("97123", CommonFields.NEGATIVE_TESTS): "none;missing_neg",
        ("97123", CommonFields.POSITIVE_TESTS): "none",
    }


def test_fix_missing_pos():
    df = read_csv_and_index_fips_date(
        "fips,date,negative_tests,positive_tests,total_tests,cases\n"
        "97123,2020-04-01,9,1,10,1\n"
        "97123,2020-04-02,17,,20,2\n"
        "97123,2020-04-03,26,4,30,4\n"
    ).reset_index()

    result_df, provenance = CovidCountyDataDataSource.synthesize_test_metrics(df)

    assert to_dict([CommonFields.FIPS, CommonFields.DATE], result_df) == {
        ("97123", pd.to_datetime("2020-04-01")): {
            CommonFields.NEGATIVE_TESTS: 9,
            CommonFields.POSITIVE_TESTS: 1,
            CommonFields.TOTAL_TESTS: 10,
            CommonFields.CASES: 1,
        },
        ("97123", pd.to_datetime("2020-04-02")): {
            CommonFields.NEGATIVE_TESTS: 17,
            CommonFields.POSITIVE_TESTS: 3,
            CommonFields.TOTAL_TESTS: 20,
            CommonFields.CASES: 2,
        },
        ("97123", pd.to_datetime("2020-04-03")): {
            CommonFields.NEGATIVE_TESTS: 26,
            CommonFields.POSITIVE_TESTS: 4,
            CommonFields.TOTAL_TESTS: 30,
            CommonFields.CASES: 4,
        },
    }
    assert provenance.to_dict() == {
        ("97123", CommonFields.NEGATIVE_TESTS): "none",
        ("97123", CommonFields.POSITIVE_TESTS): "none;missing_pos",
    }
