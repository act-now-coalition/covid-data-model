from typing import List, Optional
import os
import enum
import logging
import pathlib
import pandas as pd
import structlog
from structlog.stdlib import BoundLogger

from libs.us_state_abbrev import US_STATE_ABBREV

if os.getenv("COVID_DATA_PUBLIC"):
    LOCAL_PUBLIC_DATA_PATH = pathlib.Path(os.getenv("COVID_DATA_PUBLIC"))
else:
    LOCAL_PUBLIC_DATA_PATH = (
        pathlib.Path(__file__).parent.parent / ".." / ".." / "covid-data-public"
    )

_logger = logging.getLogger(__name__)


class AggregationLevel(enum.Enum):
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"


class DuplicateValuesForIndex(Exception):
    def __init__(self, index, duplicate_data):
        self.index = index
        self.data = duplicate_data
        super().__init__()


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


def build_aggregate_county_data_frame(jhu_data_source, cds_data_source):
    """Combines JHU and CDS county data."""
    data = jhu_data_source.timeseries()
    jhu_usa_data = data.get_data(AggregationLevel.COUNTY, country="USA", after="2020-03-01")

    data = cds_data_source.timeseries()
    cds_usa_data = data.get_data(AggregationLevel.COUNTY, country="USA", after="2020-03-01")

    # TODO(chris): Better handling of counties that are not consistent.
    # Can we move this logic to combined_datasets?

    # Before 3-22, CDS has mostly consistent county level numbers - except for
    # 3-12, where there are no numbers reported. Still need to fill that in.
    return pd.concat(
        [
            cds_usa_data[cds_usa_data.date < "2020-03-22"],
            jhu_usa_data[jhu_usa_data.date >= "2020-03-22"],
        ]
    )


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
        _logger.warning(f"{unique_fips}")

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
        _logger.warning(f"{non_matching}")
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


def _clear_common_values(log: BoundLogger, existing_df, data_source, index_fields, column_to_fill):
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


def fill_fields_and_timeseries_from_column(
    log: BoundLogger,
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    index_fields: List[str],
    date_field: str,
    column_to_fill: str,
) -> pd.DataFrame:
    """
    Return a copy of existing_df with column column_to_fill populated from new_df. Values in existing_df are copied
    to the return value except for column_to_fill of rows with index_fields present in new_df.

    If the data frames represent timeseries than pass the name of the time column in date_field. This will clear
    'column_to_fill' for all times for each index_fields in new_df. This prevents the return value containing
    timeseries with a blend of values from existing_df and new_df.

    See examples in dataset_utils_test.py

    Args:
        log: a bound structlog logger.
        existing_df: Existing data frame
        new_df: Data used to fill existing df columns
        index_fields: List of columns to use as common index.
        date_field: the time column name if the data frames represent timeseries, otherwise ''
        column_to_fill: column to add into existing_df from data_source

    Returns: Updated DataFrame with requested column filled from data_source data.
    """
    # Here is a nice tutorial on indexing:
    # https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html

    # Copy so this code can work on the data inplace without modifying the inputs.
    existing_df = existing_df.copy()
    new_df = new_df.copy()
    if column_to_fill not in existing_df.columns:
        existing_df[column_to_fill] = None

    if date_field:
        _clear_common_values(log, existing_df, new_df, index_fields, column_to_fill)
        # From here down treat the date as part of the index label for joining rows of existing_df and new_df
        index_fields.append(date_field)

    new_df.set_index(index_fields, inplace=True)
    if not existing_df.empty:
        existing_df.set_index(index_fields, inplace=True)
        common_labels = existing_df.index.intersection(new_df.index)
    else:
        # Treat an empty existing_df the same as one that has no rows in common with new_df
        common_labels = []

    missing_new_data = None
    if len(common_labels):
        # existing_df is not empty and contains labels in common with new_df. When date_field is set the date is
        # included in the compared labels and dates that are not in exsiting_df are appended later.

        # Sort suggested by 'PerformanceWarning: indexing past lexsort depth may impact performance'
        # common_labels is a sparse subset of all labels in both DataFrame and the values are looked up
        # one by one.
        existing_df.sort_index(inplace=True, sort_remaining=True)
        new_df.sort_index(inplace=True, sort_remaining=True)

        # TODO(tombrown): I have a hunch that this is mostly copying NaN values. Check and consider optimizing by
        # ignoring rows without a real value in column_to_fill.
        existing_df.loc[common_labels.values, column_to_fill] = new_df.loc[
            common_labels.values, column_to_fill
        ]
        diff = new_df.index.difference(common_labels)
        if diff.size:
            missing_new_data = new_df.loc[diff, [column_to_fill]]
    else:
        # There are no labels in common so all rows of new_df are to be appended to existing_df.
        missing_new_data = new_df.loc[:, [column_to_fill]]

    # Revert 'fips', 'state' etc back to regular columns
    existing_df.reset_index(inplace=True)
    if missing_new_data is None:
        return existing_df
    missing_new_data.reset_index(inplace=True)
    # Concat the existing data with new rows from new_data, creating a new integer index
    return pd.concat([existing_df, missing_new_data], ignore_index=True)


def fill_fields_with_data_source(
    log: BoundLogger,
    existing_df: pd.DataFrame,
    data_source: pd.DataFrame,
    index_fields: List[str],
    columns_to_fill: List[str],
) -> pd.DataFrame:
    """Pull columns from an existing data source into an existing data frame.

    Example:

        existing_df:
        ----------------
        | date | cases |
        | 4/2  | 1     |
        | 4/3  | 2     |
        | 4/4  | 3     |
        ----------------

        data_source:
        ----------------------
        | date | current_icu |
        | 4/3  | 4           |
        | 4/5  | 5           |
        ----------------------

        index_fields: ['date']

        columns_to_fill: ['current_icu']

        output:
        ------------------------------
        | date | cases | current_icu |
        | 4/2  | 1     | Na          |
        | 4/3  | 2     | 4           |
        | 4/4  | 3     | Na          |
        | 4/5  | Na    | 5           |
        ------------------------------

    Args:
        log: a bound structlog logger.
        existing_df: Existing data frame
        data_source: Data used to fill existing df columns
        index_fields: List of columns to use as common index.
        columns_to_fill: List of columns to add into existing_df from data_source

    Returns: Updated dataframe with requested columns filled from data_source data.
    """
    if len(columns_to_fill) != 1:
        raise AssertionError("Not supported")
    return fill_fields_and_timeseries_from_column(
        log, existing_df, data_source, index_fields, "", columns_to_fill[0]
    )


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
