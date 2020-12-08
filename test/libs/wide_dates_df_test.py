import io

from libs import wide_dates_df
from libs.datasets import timeseries
import temppathlib


def test_write_csv():
    ds = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,cases\n"
            "iso1:us#fips:06045,Bar County,county,2020-04-01,234\n"
            "iso1:us#fips:45123,Foo County,county,2020-04-02,456\n"
        )
    )

    expected_csv = """location_id,variable,provenance,2020-04-01,2020-04-02
iso1:us#fips:06045,cases,,234,
iso1:us#fips:45123,cases,,,456
"""
    # Call common_df.write_csv with index set to ["location_id", "date"], the expected normal index.
    with temppathlib.NamedTemporaryFile("w+") as tmp:
        wide_dates_df.write_csv(ds.timeseries_rows(), tmp.path)
        assert expected_csv == tmp.file.read()
