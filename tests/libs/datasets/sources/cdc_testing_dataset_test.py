import io

import pandas as pd
from covidactnow.datapublic import common_df

from libs.datasets.sources import cdc_testing_dataset


def test_remove_trailing_zeros():

    data_buf = io.StringIO(
        "fips,date,test_positivity_7d\n"
        f"48112,2020-08-17,0.5\n"
        f"48112,2020-08-18,0.6\n"
        f"48112,2020-08-19,0.0\n"
        f"48113,2020-08-16,0.0\n"
        f"48113,2020-08-17,0.0\n"
        f"48113,2020-08-18,0.0\n"
    )
    data = common_df.read_csv(data_buf, set_index=False)
    results = cdc_testing_dataset.remove_trailing_zeros(data)

    expected_buf = io.StringIO(
        "fips,date,test_positivity_7d\n"
        f"48112,2020-08-17,0.5\n"
        f"48112,2020-08-18,0.6\n"
        f"48112,2020-08-19,\n"
        f"48113,2020-08-16,\n"
        f"48113,2020-08-17,\n"
        f"48113,2020-08-18,\n"
    )
    expected = common_df.read_csv(expected_buf, set_index=False)
    pd.testing.assert_frame_equal(expected.sort_index(axis=1), results.sort_index(axis=1))
