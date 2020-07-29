import io

import pytest
import pandas as pd

from libs.datasets.sources import nytimes_dataset
from covidactnow.datapublic import common_df


@pytest.mark.parametrize("is_ct_county", [True, False])
def test_remove_ct_cases(is_ct_county):
    if is_ct_county:
        fips = "09001"
    else:
        fips = "36061"

    data_buf = io.StringIO(
        "fips,state,date,aggregate_level,cases\n"
        f"{fips},CT,2020-07-23,county,1000\n"
        f"{fips},CT,2020-07-24,county,1288\n"
        f"{fips},CT,2020-07-25,county,1388\n"
    )

    data = common_df.read_csv(data_buf)
    data = data.reset_index()

    results = nytimes_dataset._remove_ct_backfill_cases(data)

    if is_ct_county:
        expected_cases = pd.Series([1000, 1100, 1200], name="cases")
    else:
        expected_cases = pd.Series([1000, 1288, 1388], name="cases")

    pd.testing.assert_series_equal(expected_cases, results.cases)
