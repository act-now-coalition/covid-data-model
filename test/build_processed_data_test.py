from unittest.mock import patch

import pytest

from libs import build_processed_dataset, validate_results
from libs.datasets import CommonFields
import pandas as pd

from libs.enums import Intervention


def test_get_usa_by_states_df():
    empty_df_with_state = pd.DataFrame([], columns=['state',])
    with patch('libs.build_processed_dataset.get_state_projections_df', return_value=empty_df_with_state) as mock:
        df = build_processed_dataset.get_usa_by_states_df(input_dir="/tmp", intervention_type=Intervention.OBSERVED_INTERVENTION.value)
    validate_results.validate_states_df("MA", df)
    df.set_index([CommonFields.DATE, CommonFields.STATE], inplace=True)
    print(df.info())
    print(df.head())
    assert df.loc[("2020-04-01", "MA"), CommonFields.POSITIVE_TESTS] > 0


def test_get_testing_df_by_state():
    df = build_processed_dataset.get_testing_timeseries_by_state("MA")
    assert CommonFields.POSITIVE_TESTS in df.columns
    assert CommonFields.NEGATIVE_TESTS in df.columns
    df.set_index(CommonFields.DATE, inplace=True)
    assert df.loc["2020-04-01", CommonFields.POSITIVE_TESTS] > 0


# Check San Francisco and Houston (Harris County, TX)
@pytest.mark.parametrize(
    "fips", ["06075", "48201"],
)
def test_get_testing_by_fips(fips):
    df = build_processed_dataset.get_testing_timeseries_by_fips(fips)
    assert CommonFields.POSITIVE_TESTS in df.columns
    assert CommonFields.NEGATIVE_TESTS in df.columns
    df.set_index(CommonFields.DATE, inplace=True)
    assert df.loc["2020-04-01", CommonFields.POSITIVE_TESTS] > 0
