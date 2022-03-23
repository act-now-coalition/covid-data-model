import functools
from typing import Optional, Type
import enum
import logging
import pathlib
import pandas as pd
import structlog.stdlib
from datapublic.common_fields import CommonFields
from libs.us_state_abbrev import US_STATE_ABBREV


# Slowly changing attributes of a geographical region
GEO_DATA_COLUMNS = [
    CommonFields.FIPS,
    CommonFields.STATE,
    CommonFields.AGGREGATE_LEVEL,
    CommonFields.COUNTRY,
    CommonFields.COUNTY,
]

NON_NUMERIC_COLUMNS = GEO_DATA_COLUMNS + [
    CommonFields.CAN_LOCATION_PAGE_URL,
    CommonFields.HSA,
    CommonFields.HSA_POPULATION,
]

STATIC_INDEX_FIELDS = [
    CommonFields.AGGREGATE_LEVEL,
    CommonFields.COUNTRY,
    CommonFields.STATE,
    CommonFields.FIPS,
]

TIMESERIES_INDEX_FIELDS = [
    CommonFields.DATE,
    CommonFields.AGGREGATE_LEVEL,
    CommonFields.COUNTRY,
    CommonFields.STATE,
    CommonFields.FIPS,
]


_logger = logging.getLogger(__name__)


REPO_ROOT = pathlib.Path(__file__).parent.parent.parent

DATA_DIRECTORY = REPO_ROOT / "data"

# TODO(tom): Clean up how the wide-dates and static csv paths are passed around when removing
#  DatasetPointer. See also enum `DashboardFile` in dashboard.py.
TEST_COMBINED_WIDE_DATES_CSV_PATH = REPO_ROOT / pathlib.Path(
    "tests/data/test-combined-wide-dates.csv"
)
TEST_COMBINED_STATIC_CSV_PATH = pathlib.Path(
    str(TEST_COMBINED_WIDE_DATES_CSV_PATH).replace("-wide-dates.csv", "-static.csv")
)
MANUAL_FILTER_REMOVED_WIDE_DATES_CSV_PATH = DATA_DIRECTORY / "manual_filter_removed-wide-dates.csv"
MANUAL_FILTER_REMOVED_STATIC_CSV_PATH = DATA_DIRECTORY / "manual_filter_removed-static.csv"
COMBINED_RAW_PICKLE_GZ_PATH = DATA_DIRECTORY / "combined-raw.pkl.gz"


class AggregationLevel(enum.Enum):
    COUNTRY = "country"
    STATE = "state"
    COUNTY = "county"

    # Core Base Statistical Area
    CBSA = "cbsa"

    PLACE = "place"


class DatasetType(enum.Enum):
    MULTI_REGION = "multiregion"

    @property
    def dataset_class(self) -> Type:
        """Returns associated dataset class."""
        # Avoidling circular import errors.
        from libs.datasets import timeseries

        if self is DatasetType.MULTI_REGION:
            return timeseries.MultiRegionDataset
        else:
            raise ValueError("Bad enum")


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


def aggregate_and_get_nonmatching(data, groupby_fields, from_aggregation, to_aggregation):
    from_data = data[data.aggregate_level == from_aggregation.value]
    new_data = from_data.groupby(groupby_fields).sum().reset_index()
    new_data["aggregate_level"] = to_aggregation.value
    new_data = new_data.set_index(groupby_fields)

    existing_data = data[data.aggregate_level == to_aggregation.value]
    existing_data = existing_data.set_index(groupby_fields)

    non_matching = new_data[~new_data.index.isin(existing_data.index)]
    return non_matching


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


def make_rows_key(
    data: pd.DataFrame,
    aggregation_level: Optional[AggregationLevel] = None,
    country=None,
    fips=None,
    state=None,
    states=None,
    on=None,
    after=None,
    before=None,
    location_id_matches: Optional[str] = None,
    exclude_county_999: bool = False,
    exclude_fips_prefix: Optional[str] = None,
):
    """Create a binary array or slice selecting rows in `data` matching the given parameters."""
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
    if exclude_county_999:
        # I don't think it is possible to use the default fast eval to match a substring. Instead
        # create a binary Series here and refer to it from the query.
        not_county_999 = data[CommonFields.FIPS].str[-3:] != "999"
        query_parts.append("@not_county_999")
    if location_id_matches:
        location_id_match_mask = data.index.get_level_values(CommonFields.LOCATION_ID).str.match(
            location_id_matches
        )
        query_parts.append("@location_id_match_mask")
    if exclude_fips_prefix:
        not_fips_prefix = data[CommonFields.FIPS].str[0:2] != exclude_fips_prefix
        query_parts.append("@not_fips_prefix")

    if query_parts:
        return data.eval(" and ".join(query_parts))
    else:
        # Select all rows
        return slice(None, None, None)


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
    # Make a list of the GEO_DATA_COLUMNS in df, in GEO_DATA_COLUMNS order. The intersection method returns
    # values in a different order, which makes comparing the wide dates CSV harder.
    present_columns = [column for column in GEO_DATA_COLUMNS if column in df.columns]

    # The GEO_DATA_COLUMNS are expected to have a single value for each FIPS. Get the columns
    # from every row of each data source and then keep one of each unique row.
    all_identifiers = df.loc[:, present_columns].drop_duplicates()

    # Make a DataFrame with a unique FIPS index. If multiple rows are found with the same FIPS then there
    # are rows in the input data sources that have different values for county name, state etc.
    fips_indexed = all_identifiers.set_index(CommonFields.FIPS, verify_integrity=True)
    return fips_indexed


def build_latest_for_column(timeseries_df: pd.DataFrame, column: CommonFields) -> pd.Series:
    """Builds a series of the latest value for each column.

    Args:
        timeseries_df: Timeseries DF with location_id and date columns.
        column: Column to build latest value for.

    Returns: Series indexed by location_id with the latest value for `column`.
    """
    assert timeseries_df.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]

    data = timeseries_df.sort_index()
    return data[column].groupby([CommonFields.LOCATION_ID], sort=False).last()


@functools.lru_cache(None)
def get_geo_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIRECTORY / "geo-data.csv", dtype={CommonFields.FIPS: str}).set_index(
        CommonFields.LOCATION_ID
    )


@functools.lru_cache(None)
def get_fips_to_location() -> pd.DataFrame:
    return (
        get_geo_data()
        .reset_index()
        # location_id such as "iso1:us" don't have a FIPS. Drop them so a lookup of FIPS=NA fails.
        .dropna(subset=[CommonFields.FIPS])
        .set_index(CommonFields.FIPS)[CommonFields.LOCATION_ID]
    )
