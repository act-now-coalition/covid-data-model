import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs import wide_dates_df
from libs.datasets.timeseries import TimeseriesDataset
import temppathlib


def test_write_csv():
    df = pd.DataFrame(
        {
            CommonFields.DATE: pd.to_datetime(["2020-04-01", "2020-04-02"]),
            CommonFields.FIPS: ["06045", "45123"],
            CommonFields.CASES: [234, 456],
        }
    )
    ts = TimeseriesDataset(df)

    expected_csv = """,,summary,summary,summary,summary,summary,summary,summary,summary,summary,value,value
date,,has_value,min_date,max_date,max_value,min_value,latest_value,num_observations,largest_delta,largest_delta_date,2020-04-01 00:00:00,2020-04-02 00:00:00
fips,variable,,,,,,,,,,,
06045,cases,True,2020-04-01,2020-04-01,234,234,234,1,,,234,
45123,cases,True,2020-04-02,2020-04-02,456,456,456,1,,,,456
"""
    # Call common_df.write_csv with index set to ["fips", "date"], the expected normal index.
    with temppathlib.NamedTemporaryFile("w+") as tmp:
        wide_dates_df.write_csv(ts.get_date_columns(), tmp.path)
        assert expected_csv == tmp.file.read()
