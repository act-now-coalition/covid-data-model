import numbers
from io import StringIO
from typing import Mapping, List

import pandas as pd
import numpy as np

from datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS, CommonFields
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
import pytest


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


class NoNanDict(dict):
    """A dict that ignores None and nan values passed to __new__ and recursively creates NoNanDict for dict values."""

    # Inspired by https://stackoverflow.com/a/59685000/341400
    @staticmethod
    def is_nan(v):
        if v is None:
            return True
        if not isinstance(v, numbers.Number):
            return False
        return np.isnan(v)

    @staticmethod
    def make_value(v):
        if isinstance(v, Mapping):
            return NoNanDict(v.items())
        else:
            return v

    def __new__(cls, a):
        # Recursively apply creation of value as a NoNanDict because pandas to_dict doesn't do it for you.
        return {k: NoNanDict.make_value(v) for k, v in a if not NoNanDict.is_nan(v)}


def to_dict(keys: List[str], df: pd.DataFrame):
    """Transforms df into a dict mapping columns `keys` to a dict of the record/row in df.

    Use this to extract the values from a DataFrame for easier comparisons in assert statements.
    """
    try:
        if any(df.index.names):
            df = df.reset_index()
        df = df.set_index(keys)
        return df.to_dict(orient="index", into=NoNanDict)
    except Exception:
        # Print df to provide more context when the above code raises.
        print(f"Exception in to_dict with\n{df}")
        raise


def read_csv_and_index_fips(csv_str: str) -> pd.DataFrame:
    """Read a CSV in a str to a DataFrame and set the FIPS column as an index."""
    return pd.read_csv(
        StringIO(csv_str), dtype={CommonFields.FIPS: str}, low_memory=False,
    ).set_index(CommonFields.FIPS)


def read_csv_and_index_fips_date(csv_str: str) -> pd.DataFrame:
    """Read a CSV in a str to a DataFrame and set the FIPS and DATE columns as MultiIndex."""
    return pd.read_csv(
        StringIO(csv_str),
        parse_dates=[CommonFields.DATE],
        dtype={CommonFields.FIPS: str},
        low_memory=False,
    ).set_index(COMMON_FIELDS_TIMESERIES_KEYS)


def column_as_set(
    df: pd.DataFrame,
    column: str,
    aggregation_level,
    state=None,
    states=None,
    on=None,
    after=None,
    before=None,
    exclude_county_999=False,
    exclude_fips_prefix=None,
):
    """Return values in selected rows and column of df.

    Exists to call `make_rows_key` without listing all the parameters.
    """
    rows_key = dataset_utils.make_rows_key(
        df,
        aggregation_level,
        country=None,
        fips=None,
        state=state,
        states=states,
        on=on,
        after=after,
        before=before,
        exclude_county_999=exclude_county_999,
        exclude_fips_prefix=exclude_fips_prefix,
    )
    return set(df.loc[rows_key][column])


def test_make_binary_array():
    df = read_csv_and_index_fips_date(
        "city,county,state,fips,country,aggregate_level,date,metric\n"
        "Smithville,,ZZ,97123,USA,city,2020-03-23,smithville-march23\n"
        "New York City,,ZZ,97324,USA,city,2020-03-22,march22-nyc\n"
        "New York City,,ZZ,97324,USA,city,2020-03-24,march24-nyc\n"
        ",North County,ZZ,97001,USA,county,2020-03-23,county-metric\n"
        ",,ZZ,97,USA,state,2020-03-23,mystate\n"
        ",,,,UK,country,2020-03-23,foo\n"
    ).reset_index()

    assert column_as_set(df, "country", AggregationLevel.COUNTRY) == {"UK"}
    assert column_as_set(df, "metric", AggregationLevel.STATE) == {"mystate"}
    assert column_as_set(df, "metric", None, before="2020-03-23") == {"march22-nyc"}
    assert column_as_set(df, "metric", None, after="2020-03-23") == {"march24-nyc"}
    assert column_as_set(df, "metric", None, on="2020-03-24") == {"march24-nyc"}
    assert column_as_set(
        df, "metric", None, state="ZZ", after="2020-03-22", before="2020-03-24"
    ) == {"smithville-march23", "county-metric", "mystate",}
    assert column_as_set(
        df, "metric", None, states=["ZZ"], after="2020-03-22", before="2020-03-24"
    ) == {"smithville-march23", "county-metric", "mystate",}


def test_make_binary_array_exclude_county_999():
    df = read_csv_and_index_fips_date(
        "city,county,state,fips,country,aggregate_level,date,metric\n"
        "Smithville,,ZZ,97123,USA,city,2020-03-23,smithville-march23\n"
        ",North County,ZZ,97001,USA,county,2020-03-23,county-metric\n"
        ",Unknown County,ZZ,97999,USA,county,2020-03-23,unknown-county\n"
        ",,ZZ,97,USA,state,2020-03-23,mystate\n"
        ",,,,UK,country,2020-03-23,foo\n"
    ).reset_index()

    assert column_as_set(df, "metric", AggregationLevel.COUNTY) == {
        "county-metric",
        "unknown-county",
    }
    assert column_as_set(df, "metric", AggregationLevel.COUNTY, exclude_county_999=True) == {
        "county-metric"
    }
    assert column_as_set(df, "metric", None, exclude_county_999=True) == {
        "smithville-march23",
        "county-metric",
        "mystate",
        "foo",
    }


def test_make_binary_array_exclude_fips_prefix():
    df = read_csv_and_index_fips_date(
        "city,county,state,fips,country,aggregate_level,date,metric\n"
        "Smithville,,ZZ,97123,USA,city,2020-03-23,smithville-march23\n"
        ",North County,ZZ,01001,USA,county,2020-03-23,county-in-01\n"
        ",Unknown County,ZZ,97999,USA,county,2020-03-23,unknown-county\n"
        ",,ZZ,97,USA,state,2020-03-23,state97\n"
        ",,,,UK,country,2020-03-23,country-uk\n"
    ).reset_index()

    assert column_as_set(df, "metric", AggregationLevel.COUNTY) == {
        "county-in-01",
        "unknown-county",
    }
    assert column_as_set(df, "metric", AggregationLevel.COUNTY, exclude_fips_prefix="97") == {
        "county-in-01"
    }
    assert column_as_set(df, "metric", None, exclude_fips_prefix="01") == {
        "smithville-march23",
        "unknown-county",
        "state97",
        "country-uk",
    }


def test_build_latest_for_column_unsorted():
    df = pd.read_csv(
        StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:1,2020-10-28,10\n"
            "iso1:us#fips:1,2020-10-27,6\n"
        ),
        low_memory=False,
    ).set_index([CommonFields.LOCATION_ID, CommonFields.DATE])
    result = dataset_utils.build_latest_for_column(df, CommonFields.CASES)
    expected = pd.Series([10], index=["iso1:us#fips:1"], name="cases")
    expected.index.name = "location_id"
    pd.testing.assert_series_equal(result, expected)


def test_build_latest_for_column_missing_last_value():
    df = pd.read_csv(
        StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:1,2020-10-27,10\n"
            "iso1:us#fips:1,2020-10-28,11\n"
            "iso1:us#fips:1,2020-10-29,\n"
        ),
        low_memory=False,
    ).set_index([CommonFields.LOCATION_ID, CommonFields.DATE])
    result = dataset_utils.build_latest_for_column(df, CommonFields.CASES)
    expected = pd.Series([11.0], index=["iso1:us#fips:1"], name="cases")
    expected.index.name = "location_id"
    pd.testing.assert_series_equal(result, expected)


def test_geo_data():
    geo_data = dataset_utils.get_geo_data()
    dups = geo_data.index.duplicated()
    if dups.any():
        raise ValueError(f"Duplicated location_id:\n{geo_data.index[dups]}")
