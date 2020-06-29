from unittest.mock import patch

import pytest

from covidactnow.datapublic.common_fields import CommonFields
from libs import build_processed_dataset, validate_results
from libs.datasets import CDSDataset, CovidTrackingDataSource
import pandas as pd
from libs.datasets.results_schema import CUMULATIVE_POSITIVE_TESTS, CUMULATIVE_NEGATIVE_TESTS

from libs.enums import Intervention
from more_itertools import one


def test_get_usa_by_states_df():
    empty_df_with_state = pd.DataFrame([], columns=["state",])
    with patch(
        "libs.build_processed_dataset.get_state_projections_df", return_value=empty_df_with_state
    ):
        df = build_processed_dataset.get_usa_by_states_df(
            input_dir="/tmp", intervention=Intervention.OBSERVED_INTERVENTION
        )
    validate_results.validate_states_df("TX", df)

    tx_record = one(df.loc[df[CommonFields.STATE_FULL_NAME] == "Texas"].to_dict(orient="records"))
    assert tx_record[CUMULATIVE_POSITIVE_TESTS] > 100
    assert tx_record[CUMULATIVE_NEGATIVE_TESTS] > 100


def test_get_testing_df_by_state():
    positive_field = CovidTrackingDataSource.Fields.POSITIVE_TESTS
    negative_field = CovidTrackingDataSource.Fields.NEGATIVE_TESTS
    df = build_processed_dataset.get_testing_timeseries_by_state("MA")
    assert positive_field in df.columns
    assert negative_field in df.columns
    # TODO(tom): have build_processed_dataset return a real datetime dtype so callers don't need to parse a string.
    df[CommonFields.DATE] = pd.to_datetime(df[CommonFields.DATE], format="%m/%d/%y")
    df.set_index(CommonFields.DATE, inplace=True)
    # No joke, our only source of state-level timeseries testing data is CDSDataset and it has NaN for MA until April 6.
    assert 0 < df.at["2020-04-10", positive_field] < df.at["2020-05-01", positive_field]
    assert 0 < df.at["2020-04-10", negative_field] < df.at["2020-05-01", negative_field]


# Check some counties picked arbitrarily: San Francisco/06075 and Houston (Harris County, TX)/48201
@pytest.mark.parametrize("fips", ["06075", "48201"])
def test_get_testing_by_fips(fips):
    df = build_processed_dataset.get_testing_timeseries_by_fips(fips)

    df.reset_index(inplace=True)
    assert set(CDSDataset.TEST_FIELDS) == set(df.columns)

    # TODO(tom): have build_processed_dataset return a real datetime dtype so callers don't need to parse a string.
    df[CommonFields.DATE] = pd.to_datetime(df[CommonFields.DATE], format="%m/%d/%y")
    df.set_index("date", inplace=True)

    assert df.loc["2020-04-01", CDSDataset.Fields.CASES] > 0
