import io

from libs.datasets import statistical_areas
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
import pandas as pd

from libs.pipeline import Region


def test_load_from_local_public_data():
    agg = statistical_areas.CountyAggregator.from_local_public_data()

    assert agg.cbsa_title_map["43580"] == "Sioux City, IA-NE-SD"
    assert agg.county_map["48187"] == "41700"


def test_convert():
    ts_in = MultiRegionTimeseriesDataset.from_csv(
        io.StringIO(
            "fips,state,aggregate_level,county,m1,date,foo\n"
            "55005,ZZ,county,North County,1,2020-05-01,ab\n"
            "55005,ZZ,county,North County,2,2020-05-02,cd\n"
            "55005,ZZ,county,North County,,2020-05-03,ef\n"
            "55006,ZZ,county,South County,4,2020-05-04,gh\n"
            "55,ZZ,state,Grand State,41,2020-05-01,ij\n"
            "55,ZZ,state,Grand State,43,2020-05-03,kl\n"
        )
    )
    agg = statistical_areas.CountyAggregator(
        county_map={"55005": "10001", "55006": "10001"}, cbsa_title_map={"10001": "Stat Area 1"}
    )
    ts_out = agg.aggregate(ts_in)

    pd.testing.assert_frame_equal(
        ts_out.data.loc[~ts_out.data["fips"].isna(), ts_in.data.columns], ts_in.data
    )

    ts_cbsa = ts_out.get_one_region(Region.from_cbsa_code("10001"))
    assert ts_cbsa.data["m1"].to_list() == [7]
