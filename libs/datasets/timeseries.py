import dataclasses
import datetime
import pathlib
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List, Optional, Union, TextIO
from typing import Mapping
from typing import Set
from typing import Sequence
from typing import Tuple

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields
from pandas.core.dtypes.common import is_numeric_dtype
from typing_extensions import final

import pandas as pd
import numpy as np
import structlog
from covidactnow.datapublic import common_df
from libs import pipeline
from libs.datasets import dataset_pointer
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.dataset_utils import GEO_DATA_COLUMNS
from libs.datasets.dataset_utils import NON_NUMERIC_COLUMNS
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets import taglib
from libs.datasets.taglib import TagField
from libs.datasets.taglib import TagType
from libs.datasets.taglib import UrlStr
from libs.pipeline import Region
import pandas.core.groupby.generic
from backports.cached_property import cached_property


_log = structlog.get_logger()


# Fields used as panda MultiIndex levels when tags are represented in a pd.Series
TAG_INDEX_FIELDS = [
    TagField.LOCATION_ID,
    TagField.VARIABLE,
    TagField.TYPE,
]

ANNOTATION_TAG_TYPES = [
    TagType.CUMULATIVE_LONG_TAIL_TRUNCATED,
    TagType.CUMULATIVE_TAIL_TRUNCATED,
    TagType.ZSCORE_OUTLIER,
]


class DuplicateDataException(Exception):
    def __init__(self, message, duplicates):
        self.message = message
        self.duplicates = duplicates
        super().__init__()

    def __str__(self):
        return f"DuplicateDataException({self.message})"


class BadMultiRegionWarning(UserWarning):
    pass


class RegionLatestNotFound(IndexError):
    """Requested region's latest values not found in combined data"""

    pass


@final
@dataclass(frozen=True)
class OneRegionTimeseriesDataset:
    """A set of timeseries with values from one region."""

    region: Region

    # Do not make an assumptions about a FIPS or location_id column in the DataFrame.
    data: pd.DataFrame

    latest: Dict[str, Any]

    # A default exists for convenience in tests. Non-test code is expected to explicitly set tag.
    tag: pd.Series = dataclasses.field(
        default_factory=lambda: _EMPTY_TAG_SERIES.reset_index(CommonFields.LOCATION_ID, drop=True)
    )

    @property
    def provenance(self) -> Mapping[CommonFields, List[str]]:
        provenance_series = self.tag.loc[:, [TagType.PROVENANCE]].droplevel([TagField.TYPE])
        # https://stackoverflow.com/a/56065318
        return provenance_series.groupby(level=0).agg(list).to_dict()

    @property
    def source_url(self) -> Mapping[CommonFields, List[UrlStr]]:
        source_url_series = self.tag.loc[:, [TagType.SOURCE_URL]].droplevel([TagField.TYPE])
        # https://stackoverflow.com/a/56065318
        return source_url_series.groupby(level=0).agg(list).to_dict()

    def annotations(self, metric: FieldName) -> List[taglib.AnnotationWithDate]:
        return_value = []
        for _, row in self.tag.loc[[metric], ANNOTATION_TAG_TYPES].reset_index().iterrows():
            return_value.append(taglib.AnnotationWithDate.make(row.tag_type, content=row.content))
        return return_value

    def __post_init__(self):
        assert CommonFields.LOCATION_ID in self.data.columns
        assert CommonFields.DATE in self.data.columns
        region_count = self.data[CommonFields.LOCATION_ID].nunique()
        if region_count == 0:
            _log.warning(f"Creating {self.__class__.__name__} with zero regions")
        elif region_count != 1:
            raise ValueError("Does not have exactly one region")

        if CommonFields.DATE not in self.data.columns:
            raise ValueError("A timeseries must have a date column")

        assert self.tag.index.names == [TagField.VARIABLE, TagField.TYPE]

    @property
    def date_indexed(self) -> pd.DataFrame:
        return self.data.set_index(CommonFields.DATE)

    @property
    def empty(self):
        return self.data.empty

    def has_one_region(self) -> bool:
        return True

    def yield_records(self) -> Iterable[dict]:
        # It'd be faster to use self.data.itertuples or find a way to avoid yield_records, but that
        # needs larger changes in code calling this.
        for idx, row in self.data.iterrows():
            yield row.where(pd.notnull(row), None).to_dict()

    def get_subset(self, after=None, columns=tuple()):
        rows_key = dataset_utils.make_rows_key(self.data, after=after,)
        columns_key = list(columns) if columns else slice(None, None, None)
        return dataclasses.replace(
            self, data=self.data.loc[rows_key, columns_key].reset_index(drop=True)
        )

    def remove_padded_nans(self, columns: List[str]):
        """Returns a copy of `self`, skipping rows at the start and end where `columns` are NA"""
        return dataclasses.replace(self, data=_remove_padded_nans(self.data, columns))


def _add_location_id(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the location_id column derived from FIPS"""
    if CommonFields.LOCATION_ID in df.columns:
        raise ValueError("location_id already in DataFrame")

    df = df.copy()

    df[CommonFields.LOCATION_ID] = df[CommonFields.FIPS].apply(pipeline.fips_to_location_id)
    return df


def _add_fips_if_missing(df: pd.DataFrame):
    """Adds the FIPS column derived from location_id, inplace."""
    if CommonFields.FIPS not in df.columns:
        df[CommonFields.FIPS] = df[CommonFields.LOCATION_ID].apply(pipeline.location_id_to_fips)


def _add_state_if_missing(df: pd.DataFrame):
    """Adds the state code column if missing, in place."""
    assert CommonFields.LOCATION_ID in df.columns

    if CommonFields.STATE not in df.columns:
        df[CommonFields.STATE] = df[CommonFields.LOCATION_ID].apply(
            lambda x: Region.from_location_id(x).state
        )


def _geodata_df_to_static_attribute_df(geodata_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a DataFrame to use as static from geo data taken from timeseries CSV."""
    assert geodata_df.index.names == [None]  # [CommonFields.LOCATION_ID, CommonFields.DATE]
    deduped_values = geodata_df.drop_duplicates().set_index(CommonFields.LOCATION_ID)
    duplicates = deduped_values.index.duplicated(keep=False)
    if duplicates.any():
        _log.warning("Conflicting geo data", duplicates=deduped_values.loc[duplicates, :])
        deduped_values = deduped_values.loc[~deduped_values.index.duplicated(keep="first"), :]
    return deduped_values.sort_index()


def _merge_attributes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merges the static attributes in two DataFrame objects. Non-NA values in df2 override values
    from df1.

    The returned DataFrame has an index with the union of the LOCATION_ID column of the inputs and
    columns with the union of the other columns of the inputs.
    """
    assert df1.index.names == [None]
    assert df2.index.names == [None]
    assert CommonFields.DATE not in df1.columns
    assert CommonFields.DATE not in df2.columns

    # Get the union of all location_id and columns in the input
    all_locations = sorted(set(df1[CommonFields.LOCATION_ID]) | set(df2[CommonFields.LOCATION_ID]))
    all_columns = set(df1.columns.union(df2.columns)) - {CommonFields.LOCATION_ID}
    # Transform from a column for each metric to a row for every value. Put df2 first so
    # the duplicate dropping keeps it.
    long = pd.concat(
        [df2.melt(id_vars=[CommonFields.LOCATION_ID]), df1.melt(id_vars=[CommonFields.LOCATION_ID])]
    )
    # Drop duplicate values for the same LOCATION_ID, VARIABLE
    long_deduped = long.drop_duplicates()
    long_deduped = long_deduped.set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE])[
        PdFields.VALUE
    ].dropna()
    # If the LOCATION_ID, VARIABLE index contains duplicates then df2 is changing a value.
    # This is not expected so log a warning before dropping the old value.
    dups = long_deduped.index.duplicated(keep=False)
    if dups.any():
        # Is this worth logging?
        # _log.info(f"Regional attributes changed", changes=long_deduped.loc[dups, :].sort_index())
        long_deduped = long_deduped.loc[~long_deduped.index.duplicated(keep="first"), :]
    # Transform back to a column for each metric.
    wide = long_deduped.unstack()

    wide = wide.reindex(index=all_locations)
    missing_columns = all_columns - set(wide.columns)
    if missing_columns:
        _log.debug(f"Re-adding empty columns: {missing_columns}")
        wide = wide.reindex(columns=[*wide.columns, *missing_columns])
    # Make columns expected to be numeric have a numeric dtype so that aggregation functions
    # work on them.
    numeric_columns = list(all_columns - set(NON_NUMERIC_COLUMNS))
    wide[numeric_columns] = wide[numeric_columns].apply(pd.to_numeric, axis=0)

    assert wide.index.names == [CommonFields.LOCATION_ID]

    return wide


# An empty pd.Series with the structure expected for the tag attribute. Use this when a
# dataset does not have any tags.
_EMPTY_TAG_SERIES = pd.Series(
    [],
    name=TagField.CONTENT,
    dtype="str",
    index=pd.MultiIndex.from_tuples([], names=TAG_INDEX_FIELDS),
)


# An empty pd.DataFrame with the structure expected for the static attribute. Use this when
# a dataset does not have any static values.
_EMPTY_REGIONAL_ATTRIBUTES_DF = pd.DataFrame([], index=pd.Index([], name=CommonFields.LOCATION_ID))


# An empty DataFrame with the expected index names for a timeseries with row labels <location_id,
# variable> and column labels <date>.
_EMPTY_TIMESERIES_WIDE_DATES_DF = pd.DataFrame(
    [],
    dtype="float",
    index=pd.MultiIndex.from_tuples([], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE]),
    columns=pd.DatetimeIndex([], name=CommonFields.DATE),
)


# An empty DataFrame with the expected index names for a timeseries with row labels <location_id,
# date> and column labels <variable>. This is the structure of most CSV files in this repo as of
# Nov 2020.
_EMPTY_TIMESERIES_WIDE_VARIABLES_DF = pd.DataFrame(
    [],
    dtype="float",
    index=pd.MultiIndex.from_tuples([], names=[CommonFields.LOCATION_ID, CommonFields.DATE]),
    columns=pd.Index([], name=PdFields.VARIABLE),
)


@final
@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class MultiRegionDataset:
    """A set of timeseries and static values from any number of regions.

    While the data may be accessed directly in the attributes `timeseries`, `static` and `provenance` for
    easier future refactoring try to use (adding if not available) higher level methods that derive
    the data you need from these attributes.

    Methods named `append_...` return a new object with more regions of data. Methods named `add_...` and
    `join_...` return a new object with more data about the same regions, such as new metrics and provenance
    information.
    """

    # Timeseries metrics with float values. Each timeseries is identified by a variable name and region
    timeseries: pd.DataFrame

    # Static data, each identified by variable name and region. This includes county name,
    # state etc (GEO_DATA_COLUMNS) and metrics that change so slowly they can be
    # considered constant, such as population and hospital beds.
    static: pd.DataFrame = _EMPTY_REGIONAL_ATTRIBUTES_DF

    # A Series of tag CONTENT values having index with levels TAG_INDEX_FIELDS (LOCATION_ID,
    # VARIABLE, TYPE). Rows with identical index values may exist.
    tag: pd.Series = _EMPTY_TAG_SERIES

    @property
    def timeseries_regions(self) -> Set[Region]:
        """Returns a set of all regions in the timeseries dataset."""

        location_ids = self.timeseries.index.get_level_values(CommonFields.LOCATION_ID)
        return set(Region.from_location_id(location_id) for location_id in location_ids)

    @cached_property
    def provenance(self) -> pd.DataFrame:
        """A Series of str with a MultiIndex with names LOCATION_ID and VARIABLE"""
        provenance_tags = self.tag.loc[:, :, [TagType.PROVENANCE]]
        return provenance_tags.droplevel([TagField.TYPE]).rename(PdFields.PROVENANCE)

    @cached_property
    def _geo_data(self) -> pd.DataFrame:
        return self.static.loc[:, self.static.columns.isin(GEO_DATA_COLUMNS)]

    @cached_property
    def dataset_type(self) -> DatasetType:
        return DatasetType.MULTI_REGION

    @lru_cache(maxsize=None)
    def static_and_timeseries_latest_with_fips(self) -> pd.DataFrame:
        """Static values merged with the latest timeseries values."""
        return _merge_attributes(
            self._timeseries_latest_values().reset_index(), self.static.reset_index()
        )

    def _timeseries_long(self) -> pd.Series:
        """Returns the timeseries data in long format Series, where all values are in a single column.

        Returns: a Series with MultiIndex LOCATION_ID, DATE, VARIABLE
        """
        return (
            self.timeseries.rename_axis(columns=PdFields.VARIABLE)
            .stack(dropna=True)
            .rename(PdFields.VALUE)
            .sort_index()
        )

    @lru_cache(maxsize=None)
    def timeseries_wide_dates(self) -> pd.DataFrame:
        """Returns the timeseries in a DataFrame with LOCATION_ID, VARIABLE index and DATE columns."""
        if self.timeseries.empty:
            return _EMPTY_TIMESERIES_WIDE_DATES_DF
        timeseries_long = self._timeseries_long()
        dates = timeseries_long.index.get_level_values(CommonFields.DATE)
        if dates.empty:
            return _EMPTY_TIMESERIES_WIDE_DATES_DF
        start_date = dates.min()
        end_date = dates.max()
        date_range = pd.date_range(start=start_date, end=end_date)
        timeseries_wide = (
            timeseries_long.unstack(CommonFields.DATE)
            .reindex(columns=date_range)
            .rename_axis(columns=CommonFields.DATE)
        )
        if not timeseries_wide.columns.is_all_dates:
            raise ValueError(f"Problem with {start_date} to {end_date}... {str(self.timeseries)}")
        return timeseries_wide

    def _timeseries_latest_values(self) -> pd.DataFrame:
        """Returns the latest value for every region and metric, derived from timeseries."""
        if self.timeseries.columns.empty:
            return pd.DataFrame([], index=pd.Index([], name=CommonFields.LOCATION_ID))
        # timeseries is already sorted by DATE with the latest at the bottom.
        long = self.timeseries.stack().droplevel(CommonFields.DATE)
        # `long` has MultiIndex with LOCATION_ID and VARIABLE (added by stack). Keep only the last
        # row with each index to get the last value for each date.
        unduplicated_and_last_mask = ~long.index.duplicated(keep="last")
        return long.loc[unduplicated_and_last_mask, :].unstack()

    def latest_in_static(self, field: FieldName) -> "MultiRegionDataset":
        """Returns a new object with the latest values from timeseries 'field' copied to static."""
        latest_series = self._timeseries_latest_values()[field]
        if field in self.static.columns and self.static[field].notna().any():
            # This looks like an attempt at copying the latest timeseries values to a static field
            # which already has some values. Currently this behavior is not needed. To implement it
            # decide if you want to clear all the existing static values or have the latest
            # override the static or have the latest only copied where the static is currently NA.
            raise ValueError("Can only copy field when static is unset")
        static_copy = self.static.copy()
        static_copy[field] = latest_series
        return dataclasses.replace(self, static=static_copy)

    @staticmethod
    def from_timeseries_wide_dates_df(timeseries_wide_dates: pd.DataFrame) -> "MultiRegionDataset":
        """Make a new dataset from a DataFrame as returned by timeseries_wide_dates."""
        assert timeseries_wide_dates.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
        assert timeseries_wide_dates.columns.names == [CommonFields.DATE]
        timeseries_wide_dates.columns: pd.DatetimeIndex = pd.to_datetime(
            timeseries_wide_dates.columns
        )
        timeseries_wide_variables = (
            timeseries_wide_dates.stack().unstack(PdFields.VARIABLE).sort_index()
        )
        return MultiRegionDataset(timeseries=timeseries_wide_variables)

    @staticmethod
    def from_geodata_timeseries_df(timeseries_and_geodata_df: pd.DataFrame) -> "MultiRegionDataset":
        """Make a new dataset from a DataFrame containing timeseries (real-valued metrics) and
        static geo data (county name etc)."""
        assert timeseries_and_geodata_df.index.names == [None]
        timeseries_and_geodata_df = timeseries_and_geodata_df.set_index(
            [CommonFields.LOCATION_ID, CommonFields.DATE]
        )

        geodata_column_mask = timeseries_and_geodata_df.columns.isin(
            set(TIMESERIES_INDEX_FIELDS) | set(GEO_DATA_COLUMNS)
        )
        timeseries_df = timeseries_and_geodata_df.loc[:, ~geodata_column_mask]
        # Change all columns in timeseries_df to have a numeric dtype, as checked in __post_init__
        if timeseries_df.empty:
            # Use astype to force columns in an empty DataFrame to numeric dtypes.
            # to_numeric won't modify an empty column with dtype=object.
            timeseries_df = timeseries_df.astype(float)
        else:
            # Modify various kinds of NA (which will keep the column dtype as object) to
            # NaN, which is a valid float. Apply to_numeric to columns so that int columns
            # are not modified.
            timeseries_df = timeseries_df.fillna(np.nan).apply(pd.to_numeric).sort_index()
        geodata_df = timeseries_and_geodata_df.loc[:, geodata_column_mask]

        static_df = _geodata_df_to_static_attribute_df(
            geodata_df.reset_index().drop(columns=[CommonFields.DATE])
        )

        return MultiRegionDataset(timeseries=timeseries_df, static=static_df)

    def add_fips_static_df(self, latest_df: pd.DataFrame) -> "MultiRegionDataset":
        latest_df = _add_location_id(latest_df)
        return self.add_static_values(latest_df)

    def add_static_values(self, attributes_df: pd.DataFrame) -> "MultiRegionDataset":
        """Returns a new object with non-NA values in `latest_df` added to the static attribute."""
        combined_attributes = _merge_attributes(self.static.reset_index(), attributes_df)
        assert combined_attributes.index.names == [CommonFields.LOCATION_ID]
        return dataclasses.replace(self, static=combined_attributes)

    def add_provenance_csv(self, path_or_buf: Union[pathlib.Path, TextIO]) -> "MultiRegionDataset":
        df = pd.read_csv(path_or_buf)
        if PdFields.VALUE in df.columns:
            # Handle older CSV files that used 'value' header for provenance.
            df = df.rename(columns={PdFields.VALUE: PdFields.PROVENANCE})
        series = df.set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE])[PdFields.PROVENANCE]
        return self.add_provenance_series(series)

    def add_provenance_all(self, provenance: str) -> "MultiRegionDataset":
        """Returns a new object with given provenance string for every timeseries."""
        return self.add_provenance_series(
            pd.Series([], dtype=str, name=PdFields.PROVENANCE).reindex(
                self.timeseries_wide_dates().index, fill_value=provenance
            )
        )

    def add_provenance_series(self, provenance: pd.Series) -> "MultiRegionDataset":
        """Returns a new object containing data in self and given provenance information."""
        if not self.provenance.empty:
            raise NotImplementedError("TODO(tom): add support for merging provenance data")
        assert provenance.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
        assert isinstance(provenance, pd.Series)

        new_index_df = provenance.index.to_frame()
        new_index_df[TagField.TYPE] = TagType.PROVENANCE
        tag_additions = provenance.copy()
        tag_additions.index = pd.MultiIndex.from_frame(new_index_df)
        # Make a sorted series. The order doesn't matter and sorting makes the order depend only on
        # what is represented, not the order it appears in the input.
        tag = pd.concat([self.tag, tag_additions]).sort_index().rename(TagField.CONTENT)
        return dataclasses.replace(self, tag=tag)

    @staticmethod
    def from_csv(path_or_buf: Union[pathlib.Path, TextIO]) -> "MultiRegionDataset":
        combined_df = common_df.read_csv(path_or_buf, set_index=False)
        if CommonFields.LOCATION_ID not in combined_df.columns:
            raise ValueError("MultiRegionDataset.from_csv requires location_id column")

        # Split rows with DATE NaT into latest_df and call `from_timeseries_df` to finish the
        # construction.
        rows_with_date = combined_df[CommonFields.DATE].notna()
        timeseries_df = combined_df.loc[rows_with_date, :]

        # Extract rows of combined_df which don't have a date.
        latest_df = combined_df.loc[~rows_with_date, :]

        dataset = MultiRegionDataset.from_geodata_timeseries_df(timeseries_df)
        if not latest_df.empty:
            dataset = dataset.add_static_values(latest_df.drop(columns=[CommonFields.DATE]))

        if isinstance(path_or_buf, pathlib.Path):
            provenance_path = pathlib.Path(str(path_or_buf).replace(".csv", "-provenance.csv"))
            if provenance_path.exists():
                dataset = dataset.add_provenance_csv(provenance_path)
        return dataset

    @staticmethod
    def read_from_pointer(pointer: dataset_pointer.DatasetPointer) -> "MultiRegionDataset":
        wide_dates_df = pd.read_csv(pointer.path_wide_dates(), low_memory=False)
        wide_dates_df = wide_dates_df.set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE])

        # Iterate through all known tag types. The following are populated while iterating.
        tag_columns_mask = pd.Series(False, index=wide_dates_df.columns)
        tag_df_to_concat = []
        for tag_type in TagType:
            # Extract tag_type columns from the wide date DataFrame. This parses the column names
            # created in `timeseries_rows`.
            tag_column_mask = wide_dates_df.columns.str.match(r"\A" + tag_type + r"(-\d+)?\Z")
            if tag_column_mask.any():
                tag_columns = wide_dates_df.loc[:, tag_column_mask]
                tag_series = tag_columns.stack().reset_index(-1, drop=True).rename(TagField.CONTENT)
                tag_columns_mask = tag_columns_mask | tag_column_mask
                if not tag_series.empty:
                    tag_df = tag_series.reset_index()
                    tag_df[TagField.TYPE] = tag_type
                    tag_df_to_concat.append(tag_df)

        # Assume all columns that didn't match a tag_type are dates.
        wide_dates_df = wide_dates_df.loc[:, ~tag_columns_mask]
        wide_dates_df.columns = pd.to_datetime(wide_dates_df.columns)
        wide_dates_df = wide_dates_df.rename_axis(columns=CommonFields.DATE)

        static_df = pd.read_csv(
            pointer.path_static(), dtype={CommonFields.FIPS: str}, low_memory=False
        )

        # TODO(tom): Delete this once annotation file is retired
        if pointer.path_annotation().is_file():
            annotations = pd.read_csv(pointer.path_annotation(), low_memory=False)
            tag_df_to_concat.append(annotations)

        dataset = MultiRegionDataset.from_timeseries_wide_dates_df(wide_dates_df).add_static_values(
            static_df
        )
        if tag_df_to_concat:
            dataset = dataset.append_tag_df(pd.concat(tag_df_to_concat))

        return dataset

    @staticmethod
    def from_fips_timeseries_df(ts_df: pd.DataFrame) -> "MultiRegionDataset":
        ts_df = _add_location_id(ts_df)
        _add_state_if_missing(ts_df)

        return MultiRegionDataset.from_geodata_timeseries_df(ts_df)

    def add_fips_provenance(self, provenance):
        # Check that current index is as expected. Names will be fixed after remapping, below.
        assert provenance.index.names == [CommonFields.FIPS, PdFields.VARIABLE]
        provenance = provenance.copy()
        provenance.index = provenance.index.map(
            lambda i: (pipeline.fips_to_location_id(i[0]), i[1])
        )
        provenance.index.rename([CommonFields.LOCATION_ID, PdFields.VARIABLE], inplace=True)
        provenance.rename(PdFields.PROVENANCE, inplace=True)
        return self.add_provenance_series(provenance)

    @staticmethod
    def new_without_timeseries() -> "MultiRegionDataset":
        return MultiRegionDataset.from_fips_timeseries_df(
            pd.DataFrame([], columns=[CommonFields.FIPS, CommonFields.DATE])
        )

    def __post_init__(self):
        """Checks that attributes of this object meet certain expectations."""
        # These asserts provide runtime-checking and a single place for humans reading the code to
        # check what is expected of the attributes, beyond type.
        # timeseries.index order is important for _timeseries_latest_values correctness.
        assert self.timeseries.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
        assert self.timeseries.index.is_unique
        assert self.timeseries.index.is_monotonic_increasing
        if self.timeseries.columns.names == [None]:
            # TODO(tom): Ideally __post_init__ doesn't modify any values but tracking
            # down all the places that create a DataFrame to add a column name seems like
            # a PITA. After that is done remove this branch, leaving only the assert check.
            self.timeseries.rename_axis(columns=PdFields.VARIABLE, inplace=True)
        else:
            assert self.timeseries.columns.names == [PdFields.VARIABLE]
        numeric_columns = self.timeseries.dtypes.apply(is_numeric_dtype)
        assert numeric_columns.all()
        assert self.static.index.names == [CommonFields.LOCATION_ID]
        assert self.static.index.is_unique
        assert self.static.index.is_monotonic_increasing

        assert isinstance(self.tag, pd.Series)
        assert self.tag.index.names == TAG_INDEX_FIELDS
        # TODO(tom): Work out why is_monotonic_increasing is false (just for index with NaT
        #  and real date values?) after calling sort_index(). It may be related to
        #  https://github.com/pandas-dev/pandas/issues/35992 which is fixed in pandas 1.2.0
        # assert self.tag.index.is_monotonic_increasing
        assert self.tag.name == TagField.CONTENT
        # Check that all tag location_id are in timeseries location_id
        assert (
            self.tag.index.get_level_values(TagField.LOCATION_ID)
            .difference(self.timeseries.index.get_level_values(CommonFields.LOCATION_ID))
            .empty
        )

    def append_regions(self, other: "MultiRegionDataset") -> "MultiRegionDataset":
        common_location_id = self.static.index.intersection(other.static.index)
        if not common_location_id.empty:
            raise ValueError("Do not use append_regions with duplicate location_id")
        timeseries_df = pd.concat([self.timeseries, other.timeseries]).sort_index()
        static_df = pd.concat([self.static, other.static]).sort_index()
        tag = pd.concat([self.tag, other.tag]).sort_index()
        return MultiRegionDataset(timeseries=timeseries_df, static=static_df, tag=tag)

    def append_fips_tag_df(self, additional_tag_df: pd.DataFrame) -> "MultiRegionDataset":
        """Returns a new dataset with additional_tag_df, containing a fips column, appended."""
        additional_tag_df = _add_location_id(additional_tag_df)
        return self.append_tag_df(additional_tag_df)

    def append_tag_df(self, additional_tag_df: pd.DataFrame) -> "MultiRegionDataset":
        """Returns a new dataset with additional_tag_df appended."""
        if additional_tag_df.empty:
            return self
        # Sort by index fields, and within rows having identical index fields, by content. This
        # makes the order of values in combined_series identical, independent of the order they
        # were appended.
        combined_df = (
            self.tag.reset_index()
            .append(additional_tag_df)
            .sort_values(TAG_INDEX_FIELDS + [TagField.CONTENT])
        )
        combined_series = combined_df.set_index(TAG_INDEX_FIELDS)[TagField.CONTENT]
        return dataclasses.replace(self, tag=combined_series)

    def get_one_region(self, region: Region) -> OneRegionTimeseriesDataset:
        try:
            ts_df = self.timeseries.xs(
                region.location_id, level=CommonFields.LOCATION_ID, drop_level=False
            ).reset_index()
        except KeyError:
            ts_df = pd.DataFrame([], columns=[CommonFields.LOCATION_ID, CommonFields.DATE])
        latest_dict = self._location_id_latest_dict(region.location_id)
        if ts_df.empty and not latest_dict:
            raise RegionLatestNotFound(region)

        tag = self.tag.loc[[region.location_id]].reset_index(TagField.LOCATION_ID, drop=True)

        return OneRegionTimeseriesDataset(region=region, data=ts_df, latest=latest_dict, tag=tag)

    def _location_id_latest_dict(self, location_id: str) -> dict:
        """Returns the latest values dict of a location_id."""
        try:
            attributes_series = self.static_and_timeseries_latest_with_fips().loc[location_id, :]
        except KeyError:
            attributes_series = pd.Series([], dtype=object)
        return attributes_series.where(pd.notnull(attributes_series), None).to_dict()

    def get_regions_subset(self, regions: Collection[Region]) -> "MultiRegionDataset":
        location_ids = pd.Index(sorted(r.location_id for r in regions))
        return self.get_locations_subset(location_ids)

    def get_locations_subset(self, location_ids: Collection[str]) -> "MultiRegionDataset":
        timeseries_mask = self.timeseries.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        timeseries_df = self.timeseries.loc[timeseries_mask, :]
        static_mask = self.static.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        static_df = self.static.loc[static_mask, :]
        tag_mask = self.tag.index.get_level_values(CommonFields.LOCATION_ID).isin(location_ids)
        tag = self.tag.loc[tag_mask, :]
        return MultiRegionDataset(timeseries=timeseries_df, static=static_df, tag=tag)

    def remove_regions(self, regions: Collection[Region]) -> "MultiRegionDataset":
        location_ids = pd.Index(sorted(r.location_id for r in regions))
        return self.remove_locations(location_ids)

    def remove_locations(self, location_ids: Collection[str]) -> "MultiRegionDataset":
        timeseries_mask = self.timeseries.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        timeseries_df = self.timeseries.loc[~timeseries_mask, :]
        static_mask = self.static.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        static_df = self.static.loc[~static_mask, :]
        tag_mask = self.tag.index.get_level_values(CommonFields.LOCATION_ID).isin(location_ids)
        tag = self.tag.loc[~tag_mask, :]
        return MultiRegionDataset(timeseries=timeseries_df, static=static_df, tag=tag)

    def get_subset(
        self,
        aggregation_level: Optional[AggregationLevel] = None,
        fips: Optional[str] = None,
        state: Optional[str] = None,
        states: Optional[List[str]] = None,
        location_id_matches: Optional[str] = None,
        exclude_county_999: bool = False,
    ) -> "MultiRegionDataset":
        """Returns a new object containing data for a subset of the regions in `self`."""
        rows_key = dataset_utils.make_rows_key(
            self.static,
            aggregation_level=aggregation_level,
            fips=fips,
            state=state,
            states=states,
            location_id_matches=location_id_matches,
            exclude_county_999=exclude_county_999,
        )
        location_ids = self.static.loc[rows_key, :].index
        return self.get_locations_subset(location_ids)

    def get_counties_and_places(
        self, after: Optional[datetime.datetime] = None
    ) -> "MultiRegionDataset":
        places = self.get_subset(aggregation_level=AggregationLevel.PLACE)
        return (
            self.get_subset(aggregation_level=AggregationLevel.COUNTY)
            .append_regions(places)
            ._trim_timeseries(after=after)
        )

    def _trim_timeseries(self, *, after: datetime.datetime) -> "MultiRegionDataset":
        """Returns a new object containing only timeseries data after given date."""
        ts_rows_mask = self.timeseries.index.get_level_values(CommonFields.DATE) > after
        return dataclasses.replace(self, timeseries=self.timeseries.loc[ts_rows_mask, :])

    def groupby_region(self) -> pandas.core.groupby.generic.DataFrameGroupBy:
        return self.timeseries.groupby(CommonFields.LOCATION_ID)

    def timeseries_rows(self) -> pd.DataFrame:
        """Returns a DataFrame containing timeseries values and tag, suitable for writing
        to a CSV."""
        # Make a copy to avoid modifying the cached DataFrame
        wide_dates = self.timeseries_wide_dates().copy()
        # Format as a string here because to_csv includes a full timestamp.
        wide_dates.columns = wide_dates.columns.strftime("%Y-%m-%d")
        # When I look at the CSV I'm usually looking for the most recent values so reverse the
        # dates to put the most recent on the left.
        wide_dates = wide_dates.loc[:, wide_dates.columns[-1::-1]]
        wide_dates = wide_dates.rename_axis(None, axis="columns")

        # Each element of output_series will be a column in the returned DataFrame. There is at
        # least one column for each tag type in self.tag. If a timeseries has multiple tags with
        # the same type additional columns are added.
        output_series = []
        for tag_type, tag_series in self.tag.groupby(TagField.TYPE, sort=False):
            tag_series = tag_series.droplevel(TagField.TYPE)
            duplicates = tag_series.index.duplicated()
            output_series.append(tag_series.loc[~duplicates].rename(tag_type))
            i = 1
            while duplicates.any():
                tag_series = tag_series.loc[duplicates]
                duplicates = tag_series.index.duplicated()
                output_series.append(tag_series.loc[~duplicates].rename(f"{str(tag_type)}-{i}"))
                i += 1

        output_series.append(wide_dates)

        return pd.concat(output_series, axis=1)

    def drop_stale_timeseries(self, cutoff_date: datetime.date) -> "MultiRegionDataset":
        """Returns a new object containing only timeseries with a real value on or after cutoff_date."""
        ts = self.timeseries_wide_dates()
        recent_columns_mask = ts.columns >= cutoff_date
        recent_rows_mask = ts.loc[:, recent_columns_mask].notna().any(axis=1)
        timeseries_wide_dates = ts.loc[recent_rows_mask, :]

        # Change DataFrame with date columns to DataFrame with variable columns, similar
        # to a line in from_timeseries_wide_dates_df.
        timeseries_wide_variables = (
            timeseries_wide_dates.stack()
            .unstack(PdFields.VARIABLE)
            .reindex(columns=self.timeseries.columns)
            .sort_index()
        )
        # Only keep tag information for timeseries in the new timeseries_wide_dates.
        tag_mask = self.tag.reset_index([TagField.TYPE], drop=True).index.isin(
            timeseries_wide_dates.index
        )
        tag = self.tag.loc[tag_mask]
        return dataclasses.replace(self, timeseries=timeseries_wide_variables, tag=tag)

    def to_csv(self, path: pathlib.Path):
        """Persists timeseries to CSV.

        Args:
            path: Path to write to.
        """
        latest_data = self.static.reset_index()
        _add_fips_if_missing(latest_data)

        timeseries_data = self._geo_data.join(self.timeseries).reset_index()
        _add_fips_if_missing(timeseries_data)

        # A DataFrame with timeseries data and latest data (with DATE=NaT) together
        combined = pd.concat([timeseries_data, latest_data], ignore_index=True)
        assert combined[CommonFields.LOCATION_ID].notna().all()
        common_df.write_csv(
            combined, path, structlog.get_logger(), [CommonFields.LOCATION_ID, CommonFields.DATE]
        )
        if not self.provenance.empty:
            provenance_path = str(path).replace(".csv", "-provenance.csv")
            self.provenance.sort_index().rename(PdFields.PROVENANCE).to_csv(provenance_path)

    def write_to_dataset_pointer(self, pointer: dataset_pointer.DatasetPointer):
        """Writes `self` to files referenced by `pointer`."""
        wide_df = self.timeseries_rows()

        # 7 significant digits of precision seems like enough.
        csv_buf = wide_df.to_csv(index=True, float_format="%.7g")
        # Most timeseries don't go back to the oldest dates in the CSV so they are represented by
        # a row ending in lots of commas. Remove these because CSV readers seem to handle rows
        # with missing commas correctly.
        csv_buf = re.sub(r",+\n", r"\n", csv_buf)
        pointer.path_wide_dates().write_text(csv_buf)

        # TODO(tom) Remove once annotations file is retired.
        if pointer.path_annotation().exists():
            # annotations are not written to this file any longer so remove it to avoid duplication.
            pointer.path_annotation().unlink()

        static_sorted = common_df.index_and_sort(
            self.static, index_names=[CommonFields.LOCATION_ID], log=structlog.get_logger(),
        )
        static_sorted.to_csv(pointer.path_static())

    def drop_column_if_present(self, column: CommonFields) -> "MultiRegionDataset":
        """Drops the specified column from the timeseries if it exists"""
        return self.drop_columns_if_present([column])

    def drop_columns_if_present(self, columns: List[CommonFields]) -> "MultiRegionDataset":
        """Drops the specified columns from the timeseries if they exist"""
        timeseries_df = self.timeseries.drop(columns, axis="columns", errors="ignore")
        static_df = self.static.drop(columns, axis="columns", errors="ignore")
        tag = self.tag[~self.tag.index.get_level_values(PdFields.VARIABLE).isin(columns)]
        return MultiRegionDataset(timeseries=timeseries_df, static=static_df, tag=tag)

    def join_columns(self, other: "MultiRegionDataset") -> "MultiRegionDataset":
        """Joins the timeseries columns in `other` with those in `self`.

        Args:
            other: The timeseries dataset to join with `self`. all columns except the "geo" columns
                   will be joined into `self`.
        """
        other_non_geo_attributes = set(other.static.columns) - set(GEO_DATA_COLUMNS)
        if other_non_geo_attributes:
            raise NotImplementedError(
                f"join with other with attributes {other_non_geo_attributes} not supported"
            )
        common_ts_columns = set(other.timeseries.columns) & set(self.timeseries.columns)
        if common_ts_columns:
            # columns to be joined need to be disjoint
            raise ValueError(f"Columns are in both dataset: {common_ts_columns}")
        combined_df = pd.concat([self.timeseries, other.timeseries], axis=1)
        combined_tag = pd.concat([self.tag, other.tag]).sort_index()
        return MultiRegionDataset(timeseries=combined_df, static=self.static, tag=combined_tag)

    def iter_one_regions(self) -> Iterable[Tuple[Region, OneRegionTimeseriesDataset]]:
        """Iterates through all the regions in this object"""
        for location_id, timeseries_group in self.timeseries.groupby(
            CommonFields.LOCATION_ID, as_index=True
        ):
            latest_dict = self._location_id_latest_dict(location_id)
            region = Region.from_location_id(location_id)
            tag = self.tag.loc[[region.location_id]].reset_index(TagField.LOCATION_ID, drop=True)
            yield region, OneRegionTimeseriesDataset(
                region, timeseries_group.reset_index(), latest_dict, tag=tag
            )

    def get_county_name(self, *, region: pipeline.Region) -> str:
        return self.static.at[region.location_id, CommonFields.COUNTY]

    def provenance_map(self) -> Mapping[CommonFields, Set[str]]:
        """Returns a mapping from field name to set of provenances."""
        assert TAG_INDEX_FIELDS[2] == TagField.TYPE
        return (
            self.tag.loc[:, :, [TagType.PROVENANCE]].groupby(TagField.VARIABLE).apply(set).to_dict()
        )


def _remove_padded_nans(df, columns):
    if df[columns].isna().all(axis=None):
        return df.loc[[False] * len(df), :].reset_index(drop=True)

    first_valid_index = min(df[column].first_valid_index() for column in columns)
    last_valid_index = max(df[column].last_valid_index() for column in columns)
    df = df.iloc[first_valid_index : last_valid_index + 1]
    return df.reset_index(drop=True)


def _diff_preserving_first_value(series):
    cases = series.reset_index(CommonFields.LOCATION_ID, drop=True).loc[CommonFields.CASES, :]
    # cases is a pd.Series (a 1-D vector) with DATE index
    assert cases.index.names == [CommonFields.DATE]
    new_cases = cases.diff()
    first_date = cases.notna().idxmax()
    if pd.notna(first_date):
        new_cases[first_date] = cases[first_date]

    # Return a DataFrame so NEW_CASES is a column with DATE index.
    return pd.DataFrame({CommonFields.NEW_CASES: new_cases})


def add_new_cases(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""
    # Get timeseries data from timeseries_wide_dates because it creates a date range that includes
    # every date, even those with NA cases. This keeps the output identical when empty rows are
    # dropped or added.
    cases_wide_dates = dataset_in.timeseries_wide_dates().loc[(slice(None), CommonFields.CASES), :]
    # Calculating new cases using diff will remove the first detected value from the case series.
    # We want to capture the first day a region reports a case. Since our data sources have
    # been capturing cases in all states from the beginning of the pandemic, we are treating
    # the first day as appropriate new case data.
    # We want as_index=True so that the DataFrame returned by each _diff_preserving_first_value call
    # has the location_id added as an index before being concat-ed.
    new_cases = cases_wide_dates.groupby(CommonFields.LOCATION_ID, as_index=True, sort=False).apply(
        _diff_preserving_first_value
    )

    # Replacing days with single back tracking adjustments to be 0, reduces
    # number of na days in timeseries
    new_cases[new_cases == -1] = 0

    # Remove the occasional negative case adjustments.
    new_cases[new_cases < 0] = pd.NA
    new_cases = new_cases.dropna()

    new_cases_dataset = MultiRegionDataset(timeseries=new_cases)

    dataset_out = dataset_in.join_columns(new_cases_dataset)
    return dataset_out


def _calculate_modified_zscore(series: pd.Series, window: int = 10, min_periods=3) -> pd.Series:
    """Calculates zscore for each point in series comparing current point to past `window` days.

    Each datapoint is compared to the distribution of the past `window` days as long as there are
    `min_periods` number of non-nan values in the window.

    In the calculation of z-score, zeros are thrown out. This is done to produce better results
    for regions that regularly report zeros (for instance, RI reports zero new cases on
    each weekend day).

    Args:
        series: Series to compute statistics for.
        window: Size of window to calculate mean and std.
        min_periods: Number of periods necessary to compute a score - will return nan otherwise.

    Returns: Array of scores for each datapoint in series.
    """
    series = series.copy()
    series[series == 0] = None
    rolling_series = series.rolling(window=window, min_periods=min_periods)
    # Shifting one to exclude current datapoint
    mean = rolling_series.mean().shift(1)
    std = rolling_series.std(ddof=0).shift(1)
    z = (series - mean) / std
    return z.abs()


def drop_new_case_outliers(
    timeseries: MultiRegionDataset, zscore_threshold: float = 8.0, case_threshold: int = 30,
) -> MultiRegionDataset:
    """Identifies and drops outliers from the new case series.

    Args:
        timeseries: Timeseries.
        zscore_threshold: Z-score threshold.  All new cases with a zscore greater than the
            threshold will be removed.
        case_threshold: Min number of cases needed to count as an outlier.

    Returns: timeseries with outliers removed from new_cases.
    """
    df_copy = timeseries.timeseries.copy()
    grouped_df = timeseries.groupby_region()

    zscores = grouped_df[CommonFields.NEW_CASES].apply(_calculate_modified_zscore)
    to_exclude = (zscores > zscore_threshold) & (df_copy[CommonFields.NEW_CASES] > case_threshold)

    new_tags = taglib.TagCollection()
    # to_exclude is a Series of bools with the same index as df_copy. Iterate through the index
    # rows where to_exclude is True.
    assert to_exclude.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
    for idx, _ in to_exclude[to_exclude].iteritems():
        new_tags.add(
            taglib.ZScoreOutlier(
                date=idx[1], original_observation=df_copy.at[idx, CommonFields.NEW_CASES],
            ),
            location_id=idx[0],
            variable=CommonFields.NEW_CASES,
        )
    df_copy.loc[to_exclude, CommonFields.NEW_CASES] = np.nan

    new_timeseries = dataclasses.replace(timeseries, timeseries=df_copy).append_tag_df(
        new_tags.as_dataframe()
    )

    return new_timeseries


def drop_regions_without_population(
    mrts: MultiRegionDataset,
    known_location_id_to_drop: Sequence[str],
    log: Union[structlog.BoundLoggerBase, structlog._config.BoundLoggerLazyProxy],
) -> MultiRegionDataset:
    assert mrts.static.index.names == [CommonFields.LOCATION_ID]
    latest_population = mrts.static[CommonFields.POPULATION]
    locations_with_population = mrts.static.loc[latest_population.notna()].index
    locations_without_population = mrts.static.loc[latest_population.isna()].index
    unexpected_drops = set(locations_without_population) - set(known_location_id_to_drop)
    if unexpected_drops:
        log.warning(
            "Dropping unexpected regions without populaton", location_ids=sorted(unexpected_drops)
        )
    return mrts.get_locations_subset(locations_with_population)


def backfill_vaccination_initiated(dataset: MultiRegionDataset) -> MultiRegionDataset:
    """Backfills vaccination initiated data from total doses administered and total completed.

    Args:
        dataset: Input dataset.

    Returns: New dataset with backfilled data.
    """
    fields = [
        CommonFields.VACCINES_ADMINISTERED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
    ]
    df = dataset.timeseries_wide_dates().loc[(slice(None), fields), :]
    df_var_first = df.reorder_levels([PdFields.VARIABLE, CommonFields.LOCATION_ID])

    administered = df_var_first.loc[CommonFields.VACCINES_ADMINISTERED]
    inititiated = df_var_first.loc[[CommonFields.VACCINATIONS_INITIATED]]
    completed = df_var_first.loc[[CommonFields.VACCINATIONS_COMPLETED]]

    computed_initiated = administered - completed

    # Rename index value to be initiated
    computed_initiated = computed_initiated.rename(
        index={CommonFields.VACCINATIONS_COMPLETED: CommonFields.VACCINATIONS_INITIATED}
    )

    locations = df.index.get_level_values(CommonFields.LOCATION_ID).unique()
    locations_with_initiated = (
        inititiated.notna()
        .any(axis="columns")
        .index.get_level_values(CommonFields.LOCATION_ID)
        .unique()
    )
    locations_without_initiated = locations.difference(locations_with_initiated)

    combined_initiated_df = pd.concat(
        [
            inititiated.loc[
                (slice(CommonFields.VACCINATIONS_INITIATED), locations_with_initiated), :
            ],
            computed_initiated.loc[
                (slice(CommonFields.VACCINATIONS_INITIATED), locations_without_initiated), :
            ],
        ]
    )

    combined_initiated_df = combined_initiated_df.reorder_levels(
        [CommonFields.LOCATION_ID, PdFields.VARIABLE]
    )
    initiated_dataset = MultiRegionDataset.from_timeseries_wide_dates_df(combined_initiated_df)

    # locations that are na for all vaccinations initiated
    timeseries_copy = dataset.timeseries.copy()
    timeseries_copy.loc[:, CommonFields.VACCINATIONS_INITIATED] = initiated_dataset.timeseries.loc[
        :, CommonFields.VACCINATIONS_INITIATED
    ]

    return dataclasses.replace(dataset, timeseries=timeseries_copy)


# Column for the aggregated location_id
LOCATION_ID_AGG = "location_id_agg"


@dataclass(frozen=True)
class StaticWeightedAverageAggregation:
    """Represents an an average of `field` with static weights in `scale_field`."""

    # field/column/metric that gets aggregated using a weighted average
    field: FieldName
    # static field that used to produce the weights
    scale_factor: FieldName


WEIGHTED_AGGREGATIONS = (
    # Maybe test_positivity is better averaged using time-varying total tests, but it isn't
    # implemented. See TODO next to call to _find_scale_factors.
    StaticWeightedAverageAggregation(CommonFields.TEST_POSITIVITY, CommonFields.POPULATION),
    StaticWeightedAverageAggregation(CommonFields.TEST_POSITIVITY_7D, CommonFields.POPULATION),
    StaticWeightedAverageAggregation(CommonFields.TEST_POSITIVITY_14D, CommonFields.POPULATION),
    StaticWeightedAverageAggregation(
        CommonFields.VACCINATIONS_INITIATED_PCT, CommonFields.POPULATION
    ),
    StaticWeightedAverageAggregation(
        CommonFields.VACCINATIONS_COMPLETED_PCT, CommonFields.POPULATION
    ),
    StaticWeightedAverageAggregation(
        CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE, CommonFields.MAX_BED_COUNT
    ),
    StaticWeightedAverageAggregation(
        CommonFields.ICU_TYPICAL_OCCUPANCY_RATE, CommonFields.ICU_BEDS
    ),
)


def _apply_scaling_factor(
    df_in: pd.DataFrame,
    scale_factors: pd.DataFrame,
    aggregations: Sequence[StaticWeightedAverageAggregation],
) -> pd.DataFrame:
    """Returns a copy of df_in with some fields scaled according to `aggregations`.

    Args:
        df_in: Input un-aggregated timeseries or static data
        scale_factors: For each scale_factor field, the per-region scaling factor
        aggregations: Describes the fields to be scaled
        """
    assert df_in.index.names in (
        [CommonFields.LOCATION_ID, CommonFields.DATE],
        [CommonFields.LOCATION_ID],
    )

    # Scaled fields are modified in-place
    df_out = df_in.copy()

    for agg in aggregations:
        if agg.field in df_in.columns and agg.scale_factor in scale_factors.columns:
            df_out[agg.field] = df_out[agg.field] * scale_factors[agg.scale_factor]

    return df_out


def _find_scale_factors(
    aggregations: Sequence[StaticWeightedAverageAggregation],
    location_id_map: Mapping[str, str],
    static_agg: pd.DataFrame,
    static_in: pd.DataFrame,
    location_ids: Sequence[str],
) -> pd.DataFrame:
    assert static_in.index.names == [CommonFields.LOCATION_ID]
    assert static_agg.index.names == [CommonFields.LOCATION_ID]
    # For each location_id, calculate the scaling factor from the static data.
    scale_factors = pd.DataFrame([], index=pd.Index(location_ids).unique().sort_values())
    for scale_factor_field in {agg.scale_factor for agg in aggregations}:
        if scale_factor_field in static_in.columns and scale_factor_field in static_agg.columns:
            # Make a series with index of the un-aggregated location_ids that has values of the
            # corresponding aggregated field value.
            agg_values = (
                static_in.index.to_series(index=static_in.index)
                .map(location_id_map)  # Maps from un-aggregated to aggregated location_id
                .map(static_agg[scale_factor_field])  # Gets the aggregated value
            )
            scale_factors[scale_factor_field] = static_in[scale_factor_field] / agg_values
    return scale_factors


def _calculate_weighted_reporting_ratio(
    long_all_values: pd.Series,
    location_id_map: Mapping[str, str],
    scale_series: pd.Series,
    groupby_columns: List[str],
) -> pd.Series:
    """Calculates weighted ratio of locations reporting data scaled by `scale_series`.

    Args:
        long_all_values: All values as a series
        location_id_map: Map of input region location_id to aggregate region location id.
        scale_series: Series with index of CommonFields.LOCATION_ID of weights for each
            location id. For example, population of each location id.
        groupby_columns: Columns to group scaled values by when aggregating.

    Returns: Series of scaled ratio of regions reporting with an index of `groupby_columns`.
    """
    assert long_all_values.index.names == [CommonFields.LOCATION_ID] + groupby_columns
    assert scale_series.index.names == [CommonFields.LOCATION_ID]

    scale_field = "scale"

    location_id_df = pd.DataFrame(
        location_id_map.items(), columns=[CommonFields.LOCATION_ID, LOCATION_ID_AGG]
    ).set_index(CommonFields.LOCATION_ID)
    location_id_df[scale_field] = scale_series
    location_id_aggregated_scale_field = location_id_df.groupby(LOCATION_ID_AGG)[scale_field].sum()

    long_all_scaled_count = long_all_values.notna() * scale_series
    long_agg_scaled = long_all_scaled_count.groupby(groupby_columns).sum()
    return long_agg_scaled / location_id_aggregated_scale_field


def _aggregate_dataframe_by_region(
    df_in: pd.DataFrame,
    location_id_map: Mapping[str, str],
    *,
    reporting_ratio_location_weights: Optional[pd.Series] = None,
    reporting_ratio_required: float = 1.0,
) -> pd.DataFrame:
    """Aggregates a DataFrame using given region map. The output contains dates iff the input does."""

    if CommonFields.DATE in df_in.index.names:
        groupby_columns = [LOCATION_ID_AGG, CommonFields.DATE, PdFields.VARIABLE]
        empty_result = _EMPTY_TIMESERIES_WIDE_VARIABLES_DF
    else:
        groupby_columns = [LOCATION_ID_AGG, PdFields.VARIABLE]
        empty_result = _EMPTY_REGIONAL_ATTRIBUTES_DF

    # df_in is sometimes empty in unittests. Return a DataFrame that is also empty and
    # has enough of an index that the test passes.
    if df_in.empty:
        return empty_result

    df = df_in.copy()  # Copy because the index is modified below

    # Add a new level in the MultiIndex with the new location_id_agg
    # From https://stackoverflow.com/a/56278735
    old_idx = df.index.to_frame()
    # Add location_id_agg so that when location_id is removed the remaining MultiIndex levels
    # match the levels of groupby_columns.
    old_idx.insert(1, LOCATION_ID_AGG, old_idx[CommonFields.LOCATION_ID].map(location_id_map))
    df.index = pd.MultiIndex.from_frame(old_idx)

    # Stack into a Series with several levels in the index.
    long_all_values = df.rename_axis(columns=PdFields.VARIABLE).stack(dropna=True)
    assert long_all_values.index.names == [CommonFields.LOCATION_ID] + groupby_columns

    # Aggregate by location_id_agg, optional date and variable.
    long_agg = long_all_values.groupby(groupby_columns, sort=False).sum()

    if reporting_ratio_required:
        weighted_reporting_ratio = _calculate_weighted_reporting_ratio(
            long_all_values, location_id_map, reporting_ratio_location_weights, groupby_columns
        )
        is_valid_reporting_ratio = weighted_reporting_ratio >= reporting_ratio_required
        long_agg = long_agg.loc[is_valid_reporting_ratio]

    df_out = (
        long_agg.unstack()
        .rename_axis(index={LOCATION_ID_AGG: CommonFields.LOCATION_ID})
        .sort_index()
        .reindex(columns=df_in.columns)
    )
    assert df_in.index.names == df_out.index.names
    return df_out


def aggregate_regions(
    dataset_in: MultiRegionDataset,
    aggregate_map: Mapping[Region, Region],
    aggregations: Sequence[StaticWeightedAverageAggregation] = WEIGHTED_AGGREGATIONS,
    *,
    reporting_ratio_required_to_aggregate: Optional[float] = None,
) -> MultiRegionDataset:
    """Produces a dataset with dataset_in aggregated using sum or weighted aggregation.

    Args:
        dataset_in: Input dataset.
        aggregate_map: Region mapping input region to aggregate region.
        aggregations: Sequence of aggregation overrides to apply aggregations other
            than sum to fields.
        reporting_ratio_required_to_aggregate: Ratio of locations per aggregate region required
            to compute aggregate value for individual data points. Uses population to weight
            ratio.

    Returns: Dataset with values aggregated to aggregate regions.
    """
    assert (
        reporting_ratio_required_to_aggregate is None
        or 0 < reporting_ratio_required_to_aggregate <= 1.0
    )
    dataset_in = dataset_in.get_regions_subset(aggregate_map.keys())
    location_id_map = {
        region_in.location_id: region_agg.location_id
        for region_in, region_agg in aggregate_map.items()
    }

    scale_fields = {agg.scale_factor for agg in aggregations}
    scaled_fields = {agg.field for agg in aggregations}
    agg_common_fields = scale_fields.intersection(scaled_fields)
    # Check that a field is not both scaled and used as the scale factor. While that
    # could make sense it isn't implemented.
    if agg_common_fields:
        raise ValueError("field and scale_factor have values in common")
    # TODO(tom): Do something smarter with non-number columns in static. Currently they are
    # silently dropped. Functions such as aggregate_to_new_york_city manually copy non-number
    # columns.
    static_in = dataset_in.static.select_dtypes(include="number")
    scale_field_missing = scale_fields.difference(static_in.columns)
    if scale_field_missing:
        raise ValueError("Unable to do scaling due to missing column")
    # Split static_in into two DataFrames, by column:
    scale_fields_mask = static_in.columns.isin(scale_fields)
    # Static input values used to create scale factors and ...
    static_in_scale_fields = static_in.loc[:, scale_fields_mask]
    # ... all other static input values.
    static_in_other_fields = static_in.loc[:, ~scale_fields_mask]

    populations = None
    if reporting_ratio_required_to_aggregate is not None:
        populations = static_in.loc[:, CommonFields.POPULATION]

    static_agg_scale_fields = _aggregate_dataframe_by_region(
        static_in_scale_fields,
        location_id_map,
        reporting_ratio_location_weights=populations,
        reporting_ratio_required=reporting_ratio_required_to_aggregate,
    )
    location_ids = dataset_in.timeseries.index.get_level_values(CommonFields.LOCATION_ID)
    # TODO(tom): Add support for time-varying scale factors, for example to scale
    # test_positivity by number of tests.

    scale_factors = _find_scale_factors(
        aggregations,
        location_id_map,
        static_agg_scale_fields,
        static_in_scale_fields,
        location_ids,
    )

    static_other_fields_scaled = _apply_scaling_factor(
        static_in_other_fields, scale_factors, aggregations
    )
    timeseries_scaled = _apply_scaling_factor(dataset_in.timeseries, scale_factors, aggregations)

    static_agg_other_fields = _aggregate_dataframe_by_region(
        static_other_fields_scaled,
        location_id_map,
        reporting_ratio_location_weights=populations,
        reporting_ratio_required=reporting_ratio_required_to_aggregate,
    )
    timeseries_agg = _aggregate_dataframe_by_region(
        timeseries_scaled,
        location_id_map,
        reporting_ratio_location_weights=populations,
        reporting_ratio_required=reporting_ratio_required_to_aggregate,
    )
    static_agg = pd.concat([static_agg_scale_fields, static_agg_other_fields], axis=1)
    if static_agg.index.name != CommonFields.LOCATION_ID:
        # It looks like concat doesn't always set the index name, but haven't worked out
        # the pattern of when the fix is needed.
        static_agg = static_agg.rename_axis(index=CommonFields.LOCATION_ID)

    if CommonFields.AGGREGATE_LEVEL in dataset_in.static.columns:
        # location_id_to_level returns an AggregationLevel enum, but we use the str in DataFrames.
        # These are not equivalent so put the `value` attribute in static_agg.
        static_agg[CommonFields.AGGREGATE_LEVEL] = (
            static_agg.index.get_level_values(CommonFields.LOCATION_ID)
            .map(pipeline.location_id_to_level)
            .map(lambda l: l.value)
        )
    if CommonFields.FIPS in dataset_in.static.columns:
        static_agg[CommonFields.FIPS] = static_agg.index.get_level_values(
            CommonFields.LOCATION_ID
        ).map(pipeline.location_id_to_fips)

    # TODO(tom): Copy tags (annotations and provenance) to the return value.
    return MultiRegionDataset(timeseries=timeseries_agg, static=static_agg)


class DatasetName(str):
    """Human readable name for a dataset. In the future this may be an enum, for now it
    provides some type safety."""

    pass


def _to_datasets_wide_dates_map(
    datasets: Iterable[MultiRegionDataset],
) -> Mapping[MultiRegionDataset, pd.DataFrame]:
    """Turns an iterable of datasets to a mapping of DataFrame with identical date columns.

    The mapping depends on MultiRegionDataset being hashable by id.
    """
    datasets_wide = {ds: ds.timeseries_wide_dates() for ds in datasets}
    if not datasets_wide:
        return {}
    # Find the earliest and latest dates to make a range covering all timeseries.
    dates = pd.DatetimeIndex(
        np.hstack(
            list(df.columns.get_level_values(CommonFields.DATE) for df in datasets_wide.values())
        )
    )
    if dates.empty:
        input_date_range = pd.DatetimeIndex([], name=CommonFields.DATE)
    else:
        start_date = dates.min()
        end_date = dates.max()
        input_date_range = pd.date_range(start=start_date, end=end_date, name=CommonFields.DATE)
    datasets_wide_reindexed = {
        ds: df.reorder_levels([PdFields.VARIABLE, CommonFields.LOCATION_ID]).reindex(
            columns=input_date_range
        )
        for ds, df in datasets_wide.items()
    }
    return datasets_wide_reindexed


def combined_datasets(
    timeseries_field_datasets: Mapping[FieldName, List[MultiRegionDataset]],
    static_field_datasets: Mapping[FieldName, List[MultiRegionDataset]],
) -> MultiRegionDataset:
    """Creates a dataset that gets each field from a list of datasets.

    For each region, the timeseries from the first dataset in the list with a real value is returned.
    """
    # MultiRegionDataset in `timeseries_field_datasets` will be looked up in `datasets_wide` by id.
    datasets_wide = _to_datasets_wide_dates_map(
        chain.from_iterable(timeseries_field_datasets.values())
    )
    # TODO(tom): Consider how to factor out the timeseries and static processing. For example,
    #  create rows with the entire timeseries and tags then use groupby(location_id).first().
    #  Or maybe do something with groupby(location_id).apply if it is fast enough.
    # A list of "wide date" DataFrame (VARIABLE, LOCATION_ID index and DATE columns) that
    # will be concat-ed.
    timeseries_dfs = []
    # A list of Series that will be concat-ed
    tag_series = []
    for field, datasets in timeseries_field_datasets.items():
        # Iterate through the datasets for this field. For each dataset add location_id with data
        # in field to location_id_so_far iff the location_id is not already there.
        location_id_so_far = pd.Index([])
        for dataset in datasets:
            field_wide_df = datasets_wide[dataset].loc[[field], :]
            assert field_wide_df.index.names == [PdFields.VARIABLE, CommonFields.LOCATION_ID]
            location_ids = field_wide_df.index.get_level_values(CommonFields.LOCATION_ID)
            # Select the locations in `dataset_name` that have a timeseries for `field` and are
            # not in `location_id_so_far`.
            selected_location_id = location_ids.difference(location_id_so_far)
            timeseries_dfs.append(field_wide_df.loc[(slice(None), selected_location_id), :])
            location_id_so_far = location_id_so_far.union(selected_location_id).sort_values()
            tag_series.append(dataset.tag.loc[selected_location_id, [field]])

    static_series = []
    for field, dataset_list in static_field_datasets.items():
        static_column_so_far = None
        for dataset in dataset_list:
            dataset_column = dataset.static.get(field)
            if dataset_column is None:
                continue
            dataset_column = dataset_column.dropna()
            assert dataset_column.index.names == [CommonFields.LOCATION_ID]
            if static_column_so_far is None:
                # This is the first dataset. Copy all not-NA values of field and the location_id
                # index to static_column_so_far.
                static_column_so_far = dataset_column
            else:
                # Add to static_column_so_far values that have index labels not already in the
                # static_column_so_far.index. Thus for each location, the first dataset with a
                # value is copied and values in later dataset are not copied.
                selected_location_id = dataset_column.index.difference(static_column_so_far.index)
                static_column_so_far = pd.concat(
                    [static_column_so_far, dataset_column.loc[selected_location_id]],
                    sort=True,
                    verify_integrity=True,
                )
        static_series.append(static_column_so_far)
    if timeseries_dfs:
        output_timeseries_wide_dates = pd.concat(timeseries_dfs, verify_integrity=True)
        assert output_timeseries_wide_dates.index.names == [
            PdFields.VARIABLE,
            CommonFields.LOCATION_ID,
        ]
        assert output_timeseries_wide_dates.columns.names == [CommonFields.DATE]
        # Transform from a column for each date to a column for each variable (with rows for dates).
        # stack and unstack does the transform quickly but does not handle an empty DataFrame.
        if output_timeseries_wide_dates.empty:
            output_timeseries_wide_variables = _EMPTY_TIMESERIES_WIDE_VARIABLES_DF
        else:
            output_timeseries_wide_variables = (
                output_timeseries_wide_dates.stack().unstack(PdFields.VARIABLE).sort_index()
            )
        output_tag = pd.concat(tag_series)
    else:
        output_timeseries_wide_variables = _EMPTY_TIMESERIES_WIDE_VARIABLES_DF
        output_tag = _EMPTY_TAG_SERIES
    if static_series:
        output_static_df = pd.concat(
            static_series, axis=1, sort=True, verify_integrity=True
        ).rename_axis(index=CommonFields.LOCATION_ID)
    else:
        output_static_df = _EMPTY_REGIONAL_ATTRIBUTES_DF

    return MultiRegionDataset(
        timeseries=output_timeseries_wide_variables,
        tag=output_tag.sort_index(),
        static=output_static_df,
    )


def _aggregate_ignoring_nas(df_in: pd.DataFrame) -> Mapping:
    aggregated = {}
    for field in df_in.columns:
        if field == CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE:
            licensed_beds = df_in[CommonFields.LICENSED_BEDS]
            occupancy_rates = df_in[CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE]
            aggregated[field] = (licensed_beds * occupancy_rates).sum() / licensed_beds.sum()
        elif field == CommonFields.ICU_TYPICAL_OCCUPANCY_RATE:
            icu_beds = df_in[CommonFields.ICU_BEDS]
            occupancy_rates = df_in[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE]
            aggregated[field] = (icu_beds * occupancy_rates).sum() / icu_beds.sum()
        else:
            aggregated[field] = df_in[field].sum()
    return aggregated


def aggregate_puerto_rico_from_counties(dataset: MultiRegionDataset) -> MultiRegionDataset:
    """Returns a dataset with NA static values for the state PR aggregated from counties."""
    pr_county_mask = (dataset.static[CommonFields.STATE] == "PR") & (
        dataset.static[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.COUNTY.value
    )
    if not pr_county_mask.any():
        return dataset
    pr_counties = dataset.static.loc[pr_county_mask]
    aggregated = _aggregate_ignoring_nas(pr_counties.select_dtypes(include="number"))
    pr_location_id = pipeline.Region.from_state("PR").location_id

    patched_static = dataset.static.copy()
    for field, aggregated_value in aggregated.items():
        if pd.isna(patched_static.at[pr_location_id, field]):
            patched_static.at[pr_location_id, field] = aggregated_value

    return dataclasses.replace(dataset, static=patched_static)


def derive_vaccine_pct(ds_in: MultiRegionDataset) -> MultiRegionDataset:
    """Returns a new dataset containing everything all the input and vaccination percentage
    metrics derived from the corresponding non-percentage field where not already set."""
    field_map = {
        CommonFields.VACCINATIONS_INITIATED: CommonFields.VACCINATIONS_INITIATED_PCT,
        CommonFields.VACCINATIONS_COMPLETED: CommonFields.VACCINATIONS_COMPLETED_PCT,
    }
    ds_in_wide_dates = ds_in.timeseries_wide_dates()
    ds_in_wide_dates = ds_in_wide_dates.loc[
        ds_in_wide_dates.index.get_level_values(PdFields.VARIABLE).isin(field_map.keys())
    ]
    # TODO(tom): Preserve provenance and other tags from the original vaccination fields.
    # ds_in_wide_dates / dataset.static.loc[:, CommonFields.POPULATION] doesn't seem to align the
    # location_id correctly so be more explicit with `div`:
    derived_pct_df = (
        ds_in_wide_dates.div(
            ds_in.static.loc[:, CommonFields.POPULATION],
            level=CommonFields.LOCATION_ID,
            axis="index",
        )
        * 100.0
    )
    derived_pct_df = derived_pct_df.rename(index=field_map, level=PdFields.VARIABLE)
    # Make a dataset containing only the derived percentage metrics.
    ds_derived_pct = MultiRegionDataset.from_timeseries_wide_dates_df(derived_pct_df)

    # Combine the derived percentage with any percentages in ds_in to make a dataset containing
    # only the percentage metrics.
    # Possible optimization: Replace the combine+drop+join with a single function that copies from
    # ds_derived_pct only where a timeseries is all NaN in the original dataset.
    ds_list = [ds_in, ds_derived_pct]
    ds_all_pct = combined_datasets({field_pct: ds_list for field_pct in field_map.values()}, {})

    dataset_without_pct = ds_in.drop_columns_if_present(list(field_map.values()))
    return dataset_without_pct.join_columns(ds_all_pct)
