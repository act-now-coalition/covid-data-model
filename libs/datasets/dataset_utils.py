import enum
import logging
import pathlib
import pandas as pd
from libs import build_params

LOCAL_PUBLIC_DATA_PATH = (
    pathlib.Path(__file__).parent.parent / ".." / ".." / "covid-data-public"
)


_logger = logging.getLogger(__name__)


class AggregationLevel(enum.Enum):
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"


def strip_whitespace(data: pd.DataFrame) -> pd.DataFrame:
    """Removes all whitespace from string values.

    Note: Does not modify column names.

    Args:
        data: DataFrame

    Returns: New DataFrame with no whitespace.
    """
    # Remove all whitespace
    return data.applymap(lambda x: x.strip() if type(x) == str else x)


def parse_county_from_state(state):
    if pd.isnull(state):
        return state
    state_split = [val.strip() for val in state.split(",")]
    if len(state_split) == 2:
        return state_split[0]
    return None


def parse_state(state):
    if pd.isnull(state):
        return state
    state_split = [val.strip() for val in state.split(",")]
    state = state_split[1] if len(state_split) == 2 else state_split[0]
    return build_params.US_STATE_ABBREV.get(state, state)


def plot_grouped_data(data, group, series="source", values="cases"):
    data_by_source = data.groupby(group).sum().reset_index()
    cases_by_source = data_by_source.pivot_table(
        index=["date"], columns=series, values=values
    ).fillna(0)
    cases_by_source.plot(
        kind="bar", figsize=(15, 7), title=f"{values} by data source vs date"
    )


def build_aggregate_county_data_frame(jhu_data_source, cds_data_source):
    """Combines JHU and CDS county data."""
    data = jhu_data_source.to_generic_timeseries()
    jhu_usa_data = data.get_subset(
        AggregationLevel.COUNTY, country="USA", after="2020-03-01"
    ).data

    data = cds_data_source.to_generic_timeseries()
    cds_usa_data = data.get_subset(
        AggregationLevel.COUNTY, country="USA", after="2020-03-01"
    ).data

    # TODO(chris): Better handling of counties that are not consistent.

    # Before 3-22, CDS has mostly consistent county level numbers - except for
    # 3-12, where there are no numbers reported. Still need to fill that in.
    return pd.concat(
        [
            cds_usa_data[cds_usa_data.date < "2020-03-22"],
            jhu_usa_data[jhu_usa_data.date >= "2020-03-22"],
        ]
    )


def check_index_values_are_unique(data):
    duplicates_results = data.index.duplicated()
    duplicates = duplicates_results[duplicates_results == True]
    if len(duplicates):
        _logger.warning(f"Found {len(duplicates)} results.")


def compare_datasets(
    base, other, group, first_name="first", other_name="second", values="cases"
):
    other = other.groupby(group).sum().reset_index().set_index(group)
    base = base.groupby(group).sum().reset_index().set_index(group)
    base["info"] = first_name
    other["info"] = other_name
    common = pd.concat([base, other])
    all_combined = common.pivot_table(
        index=["date", "state"], columns="info", values=values
    ).rename_axis(None, axis=1)
    first_null = all_combined[first_name].isnull()
    first_notnull = all_combined[first_name].notnull()
    other_null = all_combined[other_name].isnull()
    other_notnull = all_combined[other_name].notnull()

    contains_both = all_combined[first_notnull & other_notnull]
    matching = contains_both[contains_both[first_name] == contains_both[other_name]]
    not_matching = contains_both[contains_both[first_name] != contains_both[other_name]]
    not_matching["delta"] = contains_both[first_name] - contains_both[other_name]
    not_matching["delta_ratio"] = (
        contains_both[first_name] - contains_both[other_name]
    ) / contains_both[first_name]
    return all_combined, matching, not_matching


def aggregate_and_get_nonmatching(
    data, groupby_fields, from_aggregation, to_aggregation
):

    from_data = data[data.aggregate_level == from_aggregation.value]
    new_data = from_data.groupby(groupby_fields).sum().reset_index()
    new_data["aggregate_level"] = to_aggregation.value
    new_data = new_data.set_index(groupby_fields)

    existing_data = data[data.aggregate_level == to_aggregation.value]
    existing_data = existing_data.set_index(groupby_fields)

    non_matching = new_data[~new_data.index.isin(existing_data.index)]
    return non_matching


def get_state_level_data(data, country, state):
    country_filter = data.country == country
    state_filter = data.state == state
    aggregation_filter = data.aggregate_level == AggregationLevel.STATE.value

    return data[country_filter & state_filter & aggregation_filter]


def get_county_level_data(data, country, state, county):
    country_filter = data.country == country
    county_filter = data.county == county
    state_filter = data.state == state
    aggregation_filter = data.aggregate_level == AggregationLevel.COUNTY.value

    return data[country_filter & state_filter & aggregation_filter & county_filter]


def build_fips_data_frame():
    from libs.datasets import FIPSPopulation

    return FIPSPopulation().data


def add_county_using_fips(data, fips_data):
    data = data.set_index("fips")
    fips_data = fips_data.set_index("fips")
    data = data.join(fips_data[["county"]], on="fips", rsuffix="_r").reset_index()

    non_matching = data[data.county.isnull() & data.fips.notnull()]

    # Not all datasources have country.  If the dataset doesn't have country,
    # assuming that data is from the us.
    if "country" in non_matching.columns:
        non_matching = non_matching[data.country == "USA"]

    if len(non_matching):
        unique_counties = sorted(non_matching.county.unique())
        _logger.warning(f"Did not match {len(unique_counties)} counties to fips data.")
        _logger.warning(f"{unique_counties}")
        # TODO: Make this an error?

    if "county_r" in data.columns:
        return data.drop("county").rename({"count_r": "county"}, axis=1)
    return data


def add_fips_using_county(data, fips_data) -> pd.Series:
    """Gets FIPS code from a data frame with a county."""
    data = data.set_index(["county", "state"])
    fips_data = fips_data.set_index(["county", "state"])
    data = data.join(
        fips_data[["fips"]], how="left", on=["county", "state"], rsuffix="_r"
    ).reset_index()

    non_matching = data[data.county.notnull() & data.fips.isnull()]

    # Not all datasources have country.  If the dataset doesn't have country,
    # assuming that data is from the us.
    if "country" in non_matching.columns:
        non_matching = non_matching[data.country == "USA"]

    if len(non_matching):
        unique_counties = sorted(non_matching.county.unique())
        _logger.warning(f"Did not match {len(unique_counties)} counties to fips data.")
        _logger.warning(f"{unique_counties}")
        # TODO: Make this an error?

    # Handles if a fips column already in the dataframe.
    if "fips_r" in data.columns:
        return data.drop("fips").rename({"fips_r": "fips"}, axis=1)

    return data
