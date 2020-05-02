import csv
import sys

from libs.datasets import CommonFields
from libs.enums import Intervention
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets.results_schema import (
    RESULT_DATA_COLUMNS_STATES,
    RESULT_DATA_COLUMNS_COUNTIES,
    EXPECTED_MISSING_STATES,
    EXPECTED_MISSING_STATES_FROM_COUNTES,
    EXPECTED_MISSING_STATES_FROM_COUNTIES_OBSERVED_INTERVENTION,
)

"""
A set of functions to validate the datasets we prepare
"""


class DataExportException(Exception):
    def __init__(self, key, message):
        self.key = key
        self.message = message


def _raise_error_if_incorrect_headers(key, headers_list, df):
    extra_df_columns = set(df.columns) - set(headers_list)
    missing_df_columns = set(headers_list) - set(df.columns)
    if extra_df_columns or missing_df_columns:
        raise DataExportException(
            key,
            "Exported headers don't match expected headers. Didn't have headers "
            f"{missing_df_columns}, had extra columns {extra_df_columns}",
        )


def _raise_error_if_not_data_from_all_states(key, df, expected_missing):
    states = set(US_STATE_ABBREV.keys())
    states_in_df = set(df[CommonFields.STATE_FULL_NAME].unique())

    missing_states_in_df = states - states_in_df - expected_missing
    extra_states_in_df = states_in_df - states
    if missing_states_in_df or extra_states_in_df:
        raise DataExportException(
            key,
            f"Missing Data from states: {missing_states_in_df}. Have extra "
            f"states in df {extra_states_in_df}",
        )


def validate_states_df(key, states_df):
    # assert the headers are what we expect
    _raise_error_if_incorrect_headers(key, RESULT_DATA_COLUMNS_STATES, states_df)

    # assert there is data from each of the states
    _raise_error_if_not_data_from_all_states(key, states_df, EXPECTED_MISSING_STATES)

    # assert no duplicated states
    if len(states_df[CommonFields.STATE_FULL_NAME].unique()) != len(
        states_df[CommonFields.STATE_FULL_NAME]
    ):
        diff = (
            states_df[CommonFields.STATE_FULL_NAME]
            - states_df[CommonFields.STATE_FULL_NAME].unique()
        )
        raise DataExportException(
            key, f"Duplicated State Data: {diff}",
        )


def validate_counties_df(key, counties_df, intervention):
    # assert the headers are what we expect
    _raise_error_if_incorrect_headers(key, RESULT_DATA_COLUMNS_COUNTIES, counties_df)

    # assert data from each of the states
    expected_missing = EXPECTED_MISSING_STATES.union(
        EXPECTED_MISSING_STATES_FROM_COUNTES
    )
    is_observed = intervention is Intervention.OBSERVED_INTERVENTION
    if is_observed:
        expected_missing = expected_missing.union(
            EXPECTED_MISSING_STATES_FROM_COUNTIES_OBSERVED_INTERVENTION
        )
    _raise_error_if_not_data_from_all_states(
        key, counties_df, expected_missing,
    )

    # assert no duplicated counties
    if len(counties_df[CommonFields.FIPS].unique()) != len(
        counties_df[CommonFields.FIPS]
    ):
        raise DataExportException(
            key,
            f"Duplicated County Data: {counties_df[counties_df.duplicated(['fips'])]['fips'] }",
        )

    # assert that the csv is a certain length
    # Note that most counties don't have inference (observed) projections yet.
    # TODO(https://github.com/covid-projections/covid-data-model/issues/371):
    # Reducing the threshold to 800 for now, but we need to verify what the
    # expected behavior is going forward.
    min_length = 250 if is_observed else 800  # 1800
    if len(counties_df["County"]) < min_length:
        raise DataExportException(
            key,
            f"Expected more counties in the output, only found {len(counties_df['County'])}",
        )


def __validate_shape_file(key, shp, shp_limit, shx, shx_limit, dbf, dbf_limit):
    # rudimentary check for this. pls tell me something better.
    shp_size = sys.getsizeof(shp)
    if shp_size < shp_limit:
        raise DataExportException(
            key, f"Expected the states shape file to be larger for {key}"
        )

    shx_size = sys.getsizeof(shx)
    if shx_size < shx_limit:
        raise DataExportException(key, f"Expected the SHX file to be larger for {key}")

    dbf_size = sys.getsizeof(dbf)
    if dbf_size < dbf_limit:
        raise DataExportException(key, f"Expected the SHX file to be larger for {key}")


def validate_states_shapefile(key, shp, shx, dbf):
    # shapefile state sizes 15434374 693 21064, from a run on April 6, 2020
    # __validate_shape_file(key, shp, 15000000, shx, 500, dbf, 20000)
    return True

def validate_counties_shapefile(key, shp, shx, dbf):
    # shapefile county sizes 90230184 17176 1097564 on April 6, 2020
    # __validate_shape_file(key, shp, 90230000, shx, 15000, dbf, 1000000)
    return True