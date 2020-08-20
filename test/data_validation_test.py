import logging
import pytest
import pandas as pd
from libs.datasets import custom_aggregations

_logger = logging.getLogger(__name__)


def default_timeseries_row(**updates):
    data = {
        "fips": "22083",
        "aggregate_level": "county",
        "date": "2020-03-26 00:00:00",
        "country": "USA",
        "state": "NY",
        "cases": 10.0,
        "deaths": 1.0,
        "recovered": 0.0,
        "current_icu": None,
        "source": "JHU",
        "county": "Richland Parish",
    }
    data.update(updates)
    return data


@pytest.mark.parametrize("are_boroughs_zero", [True, False])
def test_nyc_aggregation(are_boroughs_zero):

    nyc_county_fips = custom_aggregations.NEW_YORK_COUNTY_FIPS
    nyc_borough_fips = custom_aggregations.NYC_BOROUGH_FIPS[0]

    nyc_cases = 10
    borough_cases = 0 if are_boroughs_zero else 10
    rows = [
        default_timeseries_row(fips=nyc_county_fips, cases=nyc_cases),
        default_timeseries_row(
            fips=nyc_borough_fips,
            cases=borough_cases,
            deaths=borough_cases,
            recovered=borough_cases,
        ),
        default_timeseries_row(),
    ]

    df = pd.DataFrame(rows)

    # Todo: figure out a better way to define these groups.
    group = ["date", "source", "country", "aggregate_level", "state"]
    result = custom_aggregations.update_with_combined_new_york_counties(
        df, group, are_boroughs_zero=are_boroughs_zero
    )
    results = result.sort_values("fips").to_dict(orient="records")

    assert len(results) == 2
    nyc_result = results[1]

    if are_boroughs_zero:
        assert nyc_result["cases"] == nyc_cases
    else:
        assert nyc_result["cases"] == nyc_cases + borough_cases
        assert pd.isna(nyc_result["current_icu"])
