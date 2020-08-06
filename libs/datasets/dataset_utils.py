from typing import Optional, Type
import os
import enum
import logging
import pathlib
import pandas as pd
import structlog.stdlib
from covidactnow.datapublic.common_fields import CommonFields
from libs.us_state_abbrev import US_STATE_ABBREV


# Slowly changing attributes of a geographical region
GEO_DATA_COLUMNS = [
    CommonFields.FIPS,
    CommonFields.AGGREGATE_LEVEL,
    CommonFields.STATE,
    CommonFields.COUNTRY,
    CommonFields.COUNTY,
]


def _get_public_data_path():
    """Sets global path to covid-data-public directory."""
    if os.getenv("COVID_DATA_PUBLIC"):
        return pathlib.Path(os.getenv("COVID_DATA_PUBLIC"))

    return pathlib.Path(__file__).parent.parent / ".." / ".." / "covid-data-public"


LOCAL_PUBLIC_DATA_PATH = _get_public_data_path()


_logger = logging.getLogger(__name__)


REPO_ROOT = pathlib.Path(__file__).parent.parent.parent

DATA_DIRECTORY = REPO_ROOT / "data"


class AggregationLevel(enum.Enum):
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"


class DatasetType(enum.Enum):

    TIMESERIES = "timeseries"
    LATEST = "latest"

    @property
    def dataset_class(self) -> Type:
        """Returns associated dataset class."""
        # Avoidling circular import errors.
        from libs.datasets import timeseries

        from libs.datasets import latest_values_dataset

        if self is DatasetType.TIMESERIES:
            return timeseries.TimeseriesDataset

        if self is DatasetType.LATEST:
            return latest_values_dataset.LatestValuesDataset


class DuplicateValuesForIndex(Exception):
    def __init__(self, index, duplicate_data):
        self.index = index
        self.data = duplicate_data
        super().__init__()


def set_global_public_data_path():
    """Sets global public data path; useful if environment variable is updated after import."""
    global LOCAL_PUBLIC_DATA_PATH

    LOCAL_PUBLIC_DATA_PATH = _get_public_data_path()


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
    return US_STATE_ABBREV.get(state, state)


def plot_grouped_data(data, group, series="source", values="cases"):
    data_by_source = data.groupby(group).sum().reset_index()
    cases_by_source = data_by_source.pivot_table(
        index=["date"], columns=series, values=values
    ).fillna(0)
    cases_by_source.plot(kind="bar", figsize=(15, 7), title=f"{values} by data source vs date")


def check_index_values_are_unique(data, index=None, duplicates_as_error=True):
    """Checks index for duplicate rows.

    Args:
        data: DataFrame to check
        index: optional index to use. If not specified, uses index from `data`.
        duplicates_as_error: If True, raises an error if duplicates are found:
            otherwise logs an error.

    """
    if index:
        data = data.set_index(index)

    duplicates = data.index.duplicated()
    duplicated_data = data[duplicates]
    if sum(duplicates) and duplicates_as_error:
        raise DuplicateValuesForIndex(data.index.names, duplicated_data)
    elif sum(duplicates):
        _logger.warning(f"Found {len(duplicates)} results.")
        return duplicated_data
    return None


def compare_datasets(base, other, group, first_name="first", other_name="second", values="cases"):
    other = other.groupby(group).sum().reset_index().set_index(group)
    base = base.groupby(group).sum().reset_index().set_index(group)
    # Filling missing values
    base.loc[:, values] = base[values].fillna(0)
    other.loc[:, values] = other[values].fillna(0)

    base["info"] = first_name
    other["info"] = other_name
    common = pd.concat([base, other])
    all_combined = common.pivot_table(index=group, columns="info", values=values).rename_axis(
        None, axis=1
    )
    first_notnull = all_combined[first_name].notnull()
    other_notnull = all_combined[other_name].notnull()

    has_both = first_notnull & other_notnull
    contains_both = all_combined[first_notnull & other_notnull]
    contains_both = contains_both.reset_index()
    values_matching = contains_both[first_name] == contains_both[other_name]
    not_matching = contains_both[~values_matching]
    not_matching["delta_" + values] = contains_both[first_name] - contains_both[other_name]
    not_matching["delta_ratio_" + values] = (
        contains_both[first_name] - contains_both[other_name]
    ) / contains_both[first_name]
    matching = contains_both.loc[values_matching, :]
    missing = all_combined[~has_both & (first_notnull | other_notnull)]
    return all_combined, matching, not_matching.dropna(), missing


def aggregate_and_get_nonmatching(data, groupby_fields, from_aggregation, to_aggregation):

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


def get_county_level_data(data, country, state, county=None, fips=None):
    country_filter = data.country == country
    state_filter = data.state == state
    aggregation_filter = data.aggregate_level == AggregationLevel.COUNTY.value

    if county:
        county_filter = data.county == county
        return data[country_filter & state_filter & aggregation_filter & county_filter]
    else:
        fips_filter = data.fips == fips
        return data[country_filter & state_filter & aggregation_filter & fips_filter]


def build_fips_data_frame():
    from libs.datasets import FIPSPopulation

    return FIPSPopulation.local().data


def add_county_using_fips(data, fips_data):
    is_county = data.aggregate_level == AggregationLevel.COUNTY.value
    # Only want to add county names to county level data, so we'll slice out the county
    # data and combine it back at the end.
    not_county_df = data[~is_county]
    data = data[is_county]
    fips_data = fips_data[fips_data.aggregate_level == AggregationLevel.COUNTY.value]

    data = data.set_index(["fips", "state"])
    fips_data = fips_data.set_index(["fips", "state"])
    data = data.join(fips_data[["county"]], on=["fips", "state"], rsuffix="_r").reset_index()
    is_missing_county = data.county.isnull() & data.fips.notnull()

    data.loc[is_missing_county, "county"] = data.loc[is_missing_county, "county"].fillna("")
    non_matching = data[is_missing_county]

    # Not all datasources have country.  If the dataset doesn't have country,
    # assuming that data is from the us.
    if "country" in non_matching.columns:
        non_matching = non_matching[non_matching.country == "USA"]

    if len(non_matching):
        unique_fips = sorted(non_matching.fips.unique())
        _logger.warning(f"Did not match {len(unique_fips)} codes to county data.")
        _logger.debug(f"{unique_fips}")

    if "county_r" in data.columns:
        data = data.drop(columns="county").rename({"county_r": "county"}, axis=1)

    return pd.concat([data, not_county_df])


def assert_counties_have_fips(data, county_key="county", fips_key="fips"):
    is_county = data["aggregate_level"] == AggregationLevel.COUNTY.value
    is_fips_null = is_county & data[fips_key].isnull()
    if sum(is_fips_null):
        print(data[is_fips_null])


def add_fips_using_county(data, fips_data) -> pd.Series:
    """Gets FIPS code from a data frame with a county."""
    data = data.set_index(["county", "state"])
    original_rows = len(data)
    fips_data = fips_data.set_index(["county", "state"])
    data = data.join(
        fips_data[["fips"]], how="left", on=["county", "state"], rsuffix="_r"
    ).reset_index()

    if len(data) != original_rows:
        raise Exception("Non-unique join, check for duplicate fips data.")

    non_matching = data.loc[data.county.notnull() & data.fips.isnull(), :]

    # Not all datasources have country.  If the dataset doesn't have country,
    # assuming that data is from the us.
    if "country" in non_matching.columns:
        non_matching = non_matching.loc[data.country == "USA", :]

    if len(non_matching):
        unique_counties = sorted(non_matching.county.unique())
        _logger.warning(f"Did not match {len(unique_counties)} counties to fips data.")
        # _logger.warning(f"{non_matching}")  # This is debugging info and crowding out the stdout
        # TODO: Make this an error?

    # Handles if a fips column already in the dataframe.
    if "fips_r" in data.columns:
        return data.drop("fips").rename({"fips_r": "fips"}, axis=1)

    return data


def summarize(data, aggregate_level, groupby):
    # Standardizes the length metrics line up
    key_fmt = "{:20} {}"

    data = data[data["aggregate_level"] == aggregate_level.value]
    missing_fips = sum(data.fips.isna())
    index_size = data.groupby(groupby).size()
    non_unique = index_size > 1
    num_non_unique = sum(non_unique)
    print(key_fmt.format("Aggregate Level:", aggregate_level.value))
    print(key_fmt.format("Missing Fips Code:", missing_fips))
    print(key_fmt.format("Num non unique rows:", num_non_unique))
    print(index_size[index_size > 1])


def _clear_common_values(
    log: structlog.stdlib.BoundLogger, existing_df, data_source, index_fields, column_to_fill
):
    """For index labels shared between existing_df and data_source, clear column_to_fill in existing_df.

    existing_df is modified inplace. Index labels (the values in the index for one row) do not need to be unique in a
    table.
    """
    existing_df.set_index(index_fields, inplace=True)
    data_source.set_index(index_fields, inplace=True)
    common_labels_without_date = existing_df.index.intersection(data_source.index)
    if not common_labels_without_date.empty:
        # Maybe only do this for rows with some value in column_to_fill.
        existing_df.sort_index(inplace=True, sort_remaining=True)
        existing_df.loc[common_labels_without_date, [column_to_fill]] = None
        log.error(
            "Duplicate timeseries data",
            common_labels=common_labels_without_date.to_frame(index=False).to_dict(
                orient="records"
            ),
        )
    existing_df.reset_index(inplace=True)
    data_source.reset_index(inplace=True)


def make_binary_array(
    data: pd.DataFrame,
    aggregation_level: Optional[AggregationLevel] = None,
    country=None,
    fips=None,
    state=None,
    states=None,
    on=None,
    after=None,
    before=None,
):
    """Create a binary array selecting rows in `data` matching the given parameters."""
    query_parts = []
    # aggregation_level is almost always set. The exception is `DatasetFilter` which is used to
    # get all data in the USA, at all aggregation levels.
    if aggregation_level:
        query_parts.append(f'aggregate_level == "{aggregation_level.value}"')
    if country:
        query_parts.append("country == @country")
    if state:
        query_parts.append("state == @state")
    if fips:
        query_parts.append("fips == @fips")
    if states:
        query_parts.append("state in @states")
    if on:
        query_parts.append("date == @on")
    if after:
        query_parts.append("date > @after")
    if before:
        query_parts.append("date < @before")
    return data.eval(" and ".join(query_parts))


def fips_index_geo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame containing slowly changing attributes of each FIPS. If an attribute varies for a single
    FIPS there is a problem in the data and the set_index verify_integrity will fail.

    Args:
        df: a DataFrame that may contain data in named index and multiple rows per FIPS.

    Returns: a DataFrame with a FIPS index and one row per FIPS
    """
    if df.index.names:
        df = df.reset_index()
    # The GEO_DATA_COLUMNS are expected to have a single value for each FIPS. Get the columns
    # from every row of each data source and then keep one of each unique row.
    all_identifiers = df.loc[:, df.columns.intersection(GEO_DATA_COLUMNS)].drop_duplicates()
    # Make a DataFrame with a unique FIPS index. If multiple rows are found with the same FIPS then there
    # are rows in the input data sources that have different values for county name, state etc.
    fips_indexed = all_identifiers.set_index(CommonFields.FIPS, verify_integrity=True)
    return fips_indexed
