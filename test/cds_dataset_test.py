import numbers
from io import StringIO
from typing import Mapping, List

import pandas as pd
import numpy as np
from libs.datasets.dataset_utils import fill_fields_with_data_source
import pytest


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


def test_fill_fields_with_data_source():
    existing_df = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,current_icu,preserved\n"
            "55005,ZZ,county,North County,43,ab\n"
            "55006,ZZ,county,South County,,cd\n"
            "55,ZZ,state,Grand State,46,ef\n"
        )
    )
    new_df = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,current_icu\n"
            "55006,ZZ,county,South County,27\n"
            "55007,ZZ,county,West County,28\n"
            "55,ZZ,state,Grand State,64\n"
        )
    )

    result = fill_fields_with_data_source(
        existing_df,
        new_df,
        "fips state aggregate_level county".split(),
        ["current_icu"],
    )
    expected = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,current_icu,preserved\n"
            "55005,ZZ,county,North County,43,ab\n"
            "55006,ZZ,county,South County,27,cd\n"
            "55007,ZZ,county,West County,28,\n"
            "55,ZZ,state,Grand State,64,ef\n"
        )
    )

    assert to_dict(["fips"], result) == to_dict(["fips"], expected)


def test_fill_fields_with_data_source_timeseries():
    # Timeseries in existing_df and new_df are merged together.
    existing_df = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,cnt,date,foo\n"
            "55005,ZZ,county,North County,1,2020-05-01,ab\n"
            "55005,ZZ,county,North County,2,2020-05-02,cd\n"
            "55005,ZZ,county,North County,,2020-05-03,ef\n"
            "55006,ZZ,county,South County,4,2020-05-04,gh\n"
            "55,ZZ,state,Grand State,41,2020-05-01,ij\n"
            "55,ZZ,state,Grand State,43,2020-05-03,kl\n"
        )
    )
    new_df = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,cnt,date\n"
            "55006,ZZ,county,South County,44,2020-05-04\n"
            "55007,ZZ,county,West County,28,2020-05-03\n"
            "55005,ZZ,county,North County,3,2020-05-03\n"
            "55,ZZ,state,Grand State,42,2020-05-02\n"
        )
    )

    result = fill_fields_with_data_source(
        existing_df, new_df, "fips state aggregate_level county date".split(), ["cnt"],
    )
    expected = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,cnt,date,foo\n"
            "55005,ZZ,county,North County,1,2020-05-01,ab\n"
            "55005,ZZ,county,North County,2,2020-05-02,cd\n"
            "55005,ZZ,county,North County,3,2020-05-03,ef\n"
            "55006,ZZ,county,South County,44,2020-05-04,gh\n"
            "55007,ZZ,county,West County,28,2020-05-03,\n"
            "55,ZZ,state,Grand State,41,2020-05-01,ij\n"
            "55,ZZ,state,Grand State,42,2020-05-02,\n"
            "55,ZZ,state,Grand State,43,2020-05-03,kl\n"
        )
    )

    assert to_dict(["fips", "date"], result) == to_dict(["fips", "date"], expected)


def test_fill_fields_with_data_source_add_column():
    # existing_df does not have a current_icu column. Check that it doesn't cause a crash.
    existing_df = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,preserved\n"
            "55005,ZZ,county,North County,ab\n"
            "55,ZZ,state,Grand State,cd\n"
        )
    )
    new_df = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,current_icu\n"
            "55007,ZZ,county,West County,28\n"
            "55,ZZ,state,Grand State,64\n"
        )
    )

    result = fill_fields_with_data_source(
        existing_df,
        new_df,
        "fips state aggregate_level county".split(),
        ["current_icu"],
    )

    expected = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,current_icu,preserved\n"
            "55005,ZZ,county,North County,,ab\n"
            "55007,ZZ,county,West County,28,\n"
            "55,ZZ,state,Grand State,64,cd\n"
        )
    )
    assert to_dict(["fips"], result) == to_dict(["fips"], expected)


def test_fill_fields_with_data_source_empty_input():
    existing_df = pd.read_csv(StringIO("fips,state,aggregate_level,county,preserved\n"))
    new_df = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,current_icu\n"
            "55007,ZZ,county,West County,28\n"
            "55,ZZ,state,Grand State,64\n"
        )
    )

    result = fill_fields_with_data_source(
        existing_df,
        new_df,
        "fips state aggregate_level county".split(),
        ["current_icu"],
    )

    expected = pd.read_csv(
        StringIO(
            "fips,state,aggregate_level,county,current_icu,preserved\n"
            "55007,ZZ,county,West County,28,\n"
            "55,ZZ,state,Grand State,64,\n"
        )
    )
    assert to_dict(["fips"], result) == to_dict(["fips"], expected)
