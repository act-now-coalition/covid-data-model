from libs.datasets.dataset_utils import AggregationLevel
import pandas as pd

NEW_YORK_COUNTY = "New York County"
NEW_YORK_COUNTY_FIPS = "36061"

NYC_BOROUGH_FIPS = [
    "36047",  # Kings County
    "36081",  # Queens
    "36005",  # Bronx
    "36085",  # Richmond
]


def calculate_combined_new_york_counties(
    data, group, are_boroughs_zero=False,
):
    """Calculates combined new york metro county areas, replacing NYC with combined data.

    Most of the case data in New York City area is reported in aggregate
    but beds/population data includes values separated out by counties.

    Args:
        data: Data Frame of county level data.
        group: Group to aggregate NYC data on.  Should include all non-numeric columns,
            except fips.
        are_boroughs_zero: If true, runs a check to make sure that all borough values are
            indeed zero (for case data). If false, verifies that all boroughs have data.

    Returns: Update numbers.
    """
    all_fips_to_combine = [NEW_YORK_COUNTY_FIPS] + NYC_BOROUGH_FIPS

    is_nyc_fips = data.fips.isin(all_fips_to_combine)
    new_york_county = data[is_nyc_fips]

    # Quick integrity check to make sure we're passing in complete county.
    non_ny_county = data[data.fips.isin(NYC_BOROUGH_FIPS)]
    if are_boroughs_zero:
        grouped = non_ny_county.groupby(group).sum()
        for column in grouped.columns:
            assert sum(grouped[column]) == 0, f"{column} is unexpectedly zero."
    else:
        grouped = non_ny_county.groupby(group).sum()
        for column in grouped.columns:
            assert sum(grouped[column]) != 0, f"{column} is unexpectedly not zero."

    aggregated = new_york_county.groupby(group).sum().reset_index()
    aggregated["fips"] = NEW_YORK_COUNTY_FIPS
    aggregated["generated"] = True

    without_nyc = data[~is_nyc_fips]
    return pd.concat([without_nyc, aggregated])


def update_with_combined_new_york_counties(
    data, group, are_boroughs_zero=False,
):
    """Updates data replacing all new york county data with one number.

    """
    data = data.set_index(["aggregate_level"])
    county = data.loc[AggregationLevel.COUNTY.value].reset_index()
    county = calculate_combined_new_york_counties(
        county, group, are_boroughs_zero=are_boroughs_zero
    )

    data = data.reset_index()
    return pd.concat(
        [data[data.aggregate_level != AggregationLevel.COUNTY.value], county]
    )
