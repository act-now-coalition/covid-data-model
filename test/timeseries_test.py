from io import StringIO

from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.sources import cds_dataset
from libs.datasets.timeseries import TimeseriesDataset
from test.dataset_utils_test import to_dict
import pandas as pd
import pytest


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


def test_get_subset_and_get_data():
    input_df = pd.read_csv(
        StringIO(
            "city,county,state,fips,country,aggregate_level,date,metric\n"
            "Smithville,,ZZ,97123,USA,city,2020-03-23,smithville-march23\n"
            "New York City,,ZZ,97324,USA,city,2020-03-22,march22-nyc\n"
            "New York City,,ZZ,97324,USA,city,2020-03-24,march24-nyc\n"
            ",North County,ZZ,97001,USA,county,2020-03-23,county-metric\n"
            ",,ZZ,97,USA,state,2020-03-23,mystate\n"
            ",,XY,96,USA,state,2020-03-23,other-state\n"
            ",,,iso2:uk,UK,country,2020-03-23,you-kee\n"
            ",,,iso2:us,US,country,2020-03-23,you-ess-hey\n"
        )
    )
    ts = TimeseriesDataset(input_df)

    assert set(ts.get_subset(AggregationLevel.COUNTRY).data["metric"]) == {"you-kee", "you-ess-hey"}
    assert set(ts.get_subset(AggregationLevel.COUNTRY, country="UK").data["country"]) == {"UK"}
    assert set(ts.get_subset(AggregationLevel.STATE).data["metric"]) == {"mystate", "other-state"}
    assert set(ts.get_data(None, state="ZZ", after="2020-03-23")["metric"]) == {"march24-nyc"}
    assert set(ts.get_data(None, state="ZZ", after="2020-03-22")["metric"]) == {
        "smithville-march23",
        "county-metric",
        "mystate",
        "march24-nyc",
    }
    assert set(ts.get_data(AggregationLevel.STATE, states=["ZZ", "XY"])["metric"]) == {
        "mystate",
        "other-state",
    }
    assert set(ts.get_data(None, states=["ZZ"], on="2020-03-23")["metric"]) == {
        "smithville-march23",
        "county-metric",
        "mystate",
    }
    assert set(ts.get_data(None, states=["ZZ"], before="2020-03-23")["metric"]) == {"march22-nyc"}


def test_wide_dates():
    input_df = pd.read_csv(
        StringIO(
            "fips,county,aggregate_level,date,m1,m2\n"
            "97111,Bar County,county,2020-04-01,1,\n"
            "97111,Bar County,county,2020-04-02,2,\n"
            "97222,Foo County,county,2020-04-01,,10\n"
            "97222,Foo County,county,2020-04-03,3,30\n"
        )
    )
    ts = TimeseriesDataset(input_df)
    assert not ts.data_date_columns.empty
