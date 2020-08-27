import pytest
from covidactnow.datapublic.common_test_helpers import to_dict
from libs import pipeline
from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset
from libs.pipeline import Region
from pyseir.inference.whitelist_generator import WhitelistGenerator
from pyseir.inference.whitelist_generator import regions_in_states
from test.dataset_utils_test import read_csv_and_index_fips
from test.dataset_utils_test import read_csv_and_index_fips_date

# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.slow
def test_all_data_smoke_test():
    input_timeseries = combined_datasets.load_us_timeseries_dataset().get_subset(state="AL")
    df = WhitelistGenerator().generate_whitelist(input_timeseries)
    assert not df.empty


def test_skip_gaps_in_cases_and_deaths_metrics():
    input_df = read_csv_and_index_fips_date(
        "fips,country,state,county,aggregate_level,date,cases,deaths\n"
        "97111,US,ZZ,Bar County,county,2020-04-01,10,1\n"
        "97111,US,ZZ,Bar County,county,2020-04-02,,2\n"
        "97111,US,ZZ,Bar County,county,2020-04-03,30,\n"
        "97111,US,ZZ,Bar County,county,2020-04-04,40,4\n"
    )

    df = WhitelistGenerator().generate_whitelist(TimeseriesDataset(input_df.reset_index()))

    assert to_dict(["fips"], df) == {
        "97111": {"state": "ZZ", "county": "Bar County", "inference_ok": False},
    }


def test_inference_ok_with_5_days_cases_changed():
    # 5 days with cases data isn't enough to make inference_ok, 6 days are
    # needed so that there are 5 days with an *delta* relative to a previous day.
    input_df = read_csv_and_index_fips_date(
        "fips,country,state,county,aggregate_level,date,cases,deaths\n"
        "97111,US,ZZ,Bar County,county,2020-04-01,100,1\n"
        "97111,US,ZZ,Bar County,county,2020-04-02,200,2\n"
        "97111,US,ZZ,Bar County,county,2020-04-03,300,3\n"
        "97111,US,ZZ,Bar County,county,2020-04-04,400,4\n"
        "97111,US,ZZ,Bar County,county,2020-04-05,500,5\n"
        "97222,US,ZZ,Foo County,county,2020-04-01,100,1\n"
        "97222,US,ZZ,Foo County,county,2020-04-02,200,2\n"
        "97222,US,ZZ,Foo County,county,2020-04-03,300,3\n"
        "97222,US,ZZ,Foo County,county,2020-04-04,400,4\n"
        "97222,US,ZZ,Foo County,county,2020-04-05,500,5\n"
        "97222,US,ZZ,Foo County,county,2020-04-06,600,6\n"
    )

    df = WhitelistGenerator().generate_whitelist(TimeseriesDataset(input_df.reset_index()))

    assert to_dict(["fips"], df) == {
        "97111": {"state": "ZZ", "county": "Bar County", "inference_ok": False},
        "97222": {"state": "ZZ", "county": "Foo County", "inference_ok": True},
    }


def test_regions_in_states_basic():
    whitelist_df = read_csv_and_index_fips(
        "fips,state,county,inference_ok\n" "45111,TX,Bar County,True\n" "06222,CA,Foo County,True\n"
    ).reset_index()

    regions = regions_in_states([pipeline.Region.from_state(s) for s in ["CA", "TX"]], whitelist_df)
    assert set(regions) == {Region.from_fips("45111"), Region.from_fips("06222")}


def test_regions_in_states_fips():
    whitelist_df = read_csv_and_index_fips(
        "fips,state,county,inference_ok\n" "45111,TX,Bar County,True\n" "06222,CA,Foo County,True\n"
    ).reset_index()

    regions = regions_in_states(
        [pipeline.Region.from_state(s) for s in ["CA", "TX"]], whitelist_df, fips="45111"
    )
    assert set(regions) == {Region.from_fips("45111")}

    regions = regions_in_states(
        [pipeline.Region.from_state(s) for s in ["CA", "TX"]], whitelist_df, fips="45444"
    )
    assert set(regions) == set()
