import numbers
from io import StringIO
import pandas as pd
import numpy as np
from libs.datasets.dataset_utils import fill_fields_with_data_source
import pytest


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


class NotNanDict(dict):
    # Inspired by https://stackoverflow.com/a/59685000/341400
    @staticmethod
    def is_nan(v):
        if v is None:
            return True
        if not isinstance(v, numbers.Number):
            return False
        return np.isnan(v)

    def __new__(cls, a):
        return {k: v for k, v in a if not NotNanDict.is_nan(v)}


def to_fips_dict(df: pd.DataFrame):
    if any(df.index.names):
        df = df.reset_index()
    assert "fips" in df.columns
    df = df.set_index("fips")
    #    df = df.replace(np.nan, None)
    r = df.to_dict(orient="index", into=NotNanDict)
    return r


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

    assert to_fips_dict(result) == to_fips_dict(expected)


def test_fill_fields_with_data_source_add_column():
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
    assert to_fips_dict(result) == to_fips_dict(expected)
