import io

from libs.datasets import statistical_areas
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
import pandas as pd

from libs.datasets.timeseries import TimeseriesDataset
from libs.pipeline import Region


def test_load_from_local_public_data():
    agg = statistical_areas.CountyToCBSAAggregator.from_local_public_data()

    assert agg.cbsa_title_map["43580"] == "Sioux City, IA-NE-SD"
    assert agg.county_map["48187"] == "41700"


def test_aggregate():
    ts_in = MultiRegionTimeseriesDataset.from_timeseries(
        TimeseriesDataset.load_csv(
            io.StringIO(
                "fips,state,aggregate_level,county,m1,date,foo\n"
                "55005,ZZ,county,North County,1,2020-05-01,ab\n"
                "55005,ZZ,county,North County,2,2020-05-02,cd\n"
                "55005,ZZ,county,North County,3,2020-05-03,ef\n"
                "55006,ZZ,county,South County,3,2020-05-03,ef\n"
                "55006,ZZ,county,South County,4,2020-05-04,gh\n"
                "55,ZZ,state,Grand State,41,2020-05-01,ij\n"
                "55,ZZ,state,Grand State,43,2020-05-03,kl\n"
            )
        )
    )
    agg = statistical_areas.CountyToCBSAAggregator(
        county_map={"55005": "10001", "55006": "10001"}, cbsa_title_map={"10001": "Stat Area 1"}
    )
    ts_out = agg.aggregate(ts_in)

    assert ts_out.groupby_region().ngroups == 1

    ts_cbsa = ts_out.get_one_region(Region.from_cbsa_code("10001"))
    assert ts_cbsa.date_indexed["m1"].to_dict() == {
        pd.to_datetime("2020-05-01"): 1,
        pd.to_datetime("2020-05-02"): 2,
        pd.to_datetime("2020-05-03"): 6,
        pd.to_datetime("2020-05-04"): 4,
    }
