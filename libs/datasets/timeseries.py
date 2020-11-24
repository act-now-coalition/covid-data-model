import dataclasses
import datetime
import pathlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List, Optional, Union, TextIO
from typing import Mapping
from typing import Sequence
from typing import Tuple

from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields
from pandas.core.dtypes.common import is_numeric_dtype
from typing_extensions import final

import pandas as pd
import numpy as np
import structlog
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS
from libs import pipeline
from libs import us_state_abbrev
from libs.datasets import dataset_utils
from libs.datasets import dataset_base
from libs.datasets import custom_aggregations
from libs.datasets.common_fields import CommonIndexFields
from libs.datasets.common_fields import CommonFields
from libs.datasets.dataset_base import SaveableDatasetInterface
from libs.datasets.dataset_utils import AggregationLevel
import libs.qa.dataset_summary_gen
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.dataset_utils import GEO_DATA_COLUMNS
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.pipeline import Region
import pandas.core.groupby.generic
from backports.cached_property import cached_property

_log = structlog.get_logger()


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

    # A default exists for convience in tests. Non-test could is expected to explicitly set
    # provenance.
    # Because these objects are frozen it /might/ be safe to use default={} but using a factory to
    # make a new instance of the mutable {} is safer.
    provenance: Dict[str, str] = dataclasses.field(default_factory=dict)

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

    @property
    def date_indexed(self) -> pd.DataFrame:
        return self.data.set_index(CommonFields.DATE)

    @property
    def empty(self):
        return self.data.empty

    def has_one_region(self) -> bool:
        return True

    def yield_records(self) -> Iterable[dict]:
        # Copied from dataset_base.py
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


class TimeseriesDataset(dataset_base.DatasetBase):
    """Represents timeseries dataset.

    To make a data source compatible with the timeseries, it must have the required
    fields in the Fields class below + metrics. The other fields are generated
    in the `from_source` method.
    """

    INDEX_FIELDS = [
        CommonIndexFields.DATE,
        CommonIndexFields.AGGREGATE_LEVEL,
        CommonIndexFields.COUNTRY,
        CommonIndexFields.STATE,
        CommonIndexFields.FIPS,
    ]

    COMMON_INDEX_FIELDS = COMMON_FIELDS_TIMESERIES_KEYS

    @cached_property
    def dataset_type(self) -> DatasetType:
        return DatasetType.TIMESERIES

    @cached_property
    def empty(self):
        return self.data.empty

    def latest_values(self) -> pd.DataFrame:
        """Gets the most recent values.

        Return: DataFrame
        """
        columns_to_ffill = list(set(self.data.columns) - set(TimeseriesDataset.INDEX_FIELDS))
        data_copy = self.data.set_index([CommonFields.FIPS, CommonFields.DATE]).sort_index()
        # Groupby preserves the order of rows within a group so ffill will fill forwards by DATE.
        groups_to_fill = data_copy.groupby([CommonFields.FIPS], sort=False)
        # Consider using ffill(limit=...) to constrain how far in the past the latest values go. Currently, because
        # there may be dates missing in the index an integer limit does not provide a consistent
        # limit on the number of days in the past that may be used.
        data_copy[columns_to_ffill] = groups_to_fill[columns_to_ffill].ffill()
        return (
            data_copy.groupby([CommonFields.FIPS], sort=False)
            .last()
            # Reset FIPS back to a regular column and drop the DATE index.
            .reset_index(CommonFields.FIPS)
            .reset_index(drop=True)
        )

    def latest_values_object(self) -> LatestValuesDataset:
        return LatestValuesDataset(self.latest_values())

    def get_date_columns(self) -> pd.DataFrame:
        """Create a DataFrame with a row for each FIPS-variable timeseries and hierarchical columns.

        Returns:
            A DataFrame with a row index of COMMON_FIELDS_TIMESERIES_KEYS and columns hierarchy separating
            GEO_DATA_COLUMNS, provenance information, the timeseries summary and the complete timeseries.
        """
        ts_value_columns = (
            set(self.data.columns)
            - set(COMMON_FIELDS_TIMESERIES_KEYS)
            - set(dataset_utils.GEO_DATA_COLUMNS)
        )
        # Melt all the ts_value_columns into a single "value" column
        long = (
            self.data.loc[:, COMMON_FIELDS_TIMESERIES_KEYS + list(ts_value_columns)]
            .melt(id_vars=COMMON_FIELDS_TIMESERIES_KEYS, value_vars=ts_value_columns,)
            .dropna()
            .set_index([CommonFields.FIPS, PdFields.VARIABLE, CommonFields.DATE])
            .apply(pd.to_numeric)
        )
        # Unstack by DATE, creating a row for each FIPS-variable timeseries and a column for each DATE.
        wide_dates = long.unstack(CommonFields.DATE)
        # Drop any rows without a real value for any date.
        wide_dates = wide_dates.loc[wide_dates.loc[:, "value"].notna().any(axis=1), :]

        summary = wide_dates.loc[:, "value"].apply(
            libs.qa.dataset_summary_gen.generate_field_summary, axis=1
        )

        geo_data_per_fips = dataset_utils.fips_index_geo_data(self.data)
        # Make a DataFrame with a row for each summary.index element
        assert summary.index.names == [CommonFields.FIPS, PdFields.VARIABLE]
        geo_data = pd.merge(
            pd.DataFrame(data=[], index=summary.index),
            geo_data_per_fips,
            how="left",
            left_on=CommonFields.FIPS,  # FIPS is in the left MultiIndex
            right_index=True,
            suffixes=(False, False),
        )

        return pd.concat(
            {
                "geo_data": geo_data,
                "provenance": self.provenance,
                "summary": summary,
                "value": wide_dates["value"],
            },
            axis=1,
        )

    def get_subset(
        self,
        aggregation_level=None,
        country=None,
        fips: Optional[str] = None,
        state: Optional[str] = None,
        states: Optional[List[str]] = None,
        on: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        columns: Sequence[str] = tuple(),
    ) -> "TimeseriesDataset":
        """Fetch a new TimeseriesDataset with a subset of the data in `self`.

        Some parameters are only used in ipython notebooks."""
        rows_key = dataset_utils.make_rows_key(
            self.data,
            aggregation_level=aggregation_level,
            country=country,
            fips=fips,
            state=state,
            states=states,
            on=on,
            after=after,
            before=before,
        )
        columns_key = list(columns) if columns else slice(None, None, None)
        return self.__class__(self.data.loc[rows_key, columns_key].reset_index(drop=True))

    @classmethod
    def from_source(
        cls, source: "DataSource", fill_missing_state: bool = True
    ) -> "TimeseriesDataset":
        """Loads data from a specific datasource.

        Args:
            source: DataSource to standardize for timeseries dataset
            fill_missing_state: If True, backfills missing state data by
                calculating county level aggregates.

        Returns: Timeseries object.
        """
        data = source.data
        group = [
            CommonFields.DATE,
            CommonFields.COUNTRY,
            CommonFields.AGGREGATE_LEVEL,
            CommonFields.STATE,
        ]
        data = custom_aggregations.update_with_combined_new_york_counties(
            data, group, are_boroughs_zero=source.HAS_AGGREGATED_NYC_BOROUGH
        )

        if fill_missing_state:
            state_groupby_fields = [
                CommonFields.DATE,
                CommonFields.COUNTRY,
                CommonFields.STATE,
            ]
            non_matching = dataset_utils.aggregate_and_get_nonmatching(
                data, state_groupby_fields, AggregationLevel.COUNTY, AggregationLevel.STATE,
            ).reset_index()
            data = pd.concat([data, non_matching])

        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_county_using_fips(data, fips_data)
        is_state = data[CommonFields.AGGREGATE_LEVEL] == AggregationLevel.STATE.value
        state_fips = data.loc[is_state, CommonFields.STATE].map(us_state_abbrev.ABBREV_US_FIPS)
        data.loc[is_state, CommonFields.FIPS] = state_fips

        no_fips = data[CommonFields.FIPS].isnull()
        if no_fips.any():
            _log.warning(
                "Dropping rows without FIPS", source=str(source), rows=repr(data.loc[no_fips])
            )
            data = data.loc[~no_fips]

        dups = data.duplicated(COMMON_FIELDS_TIMESERIES_KEYS, keep=False)
        if dups.any():
            raise DuplicateDataException(f"Duplicates in {source}", data.loc[dups])

        # Choosing to sort by date
        data = data.sort_values(CommonFields.DATE)
        return cls(data, provenance=source.provenance)

    def summarize(self):
        dataset_utils.summarize(
            self.data,
            AggregationLevel.COUNTY,
            [CommonFields.DATE, CommonFields.COUNTRY, CommonFields.STATE, CommonFields.FIPS,],
        )

        dataset_utils.summarize(
            self.data,
            AggregationLevel.STATE,
            [CommonFields.DATE, CommonFields.COUNTRY, CommonFields.STATE],
        )

    @classmethod
    def load_csv(cls, path_or_buf: Union[pathlib.Path, TextIO]):
        df = common_df.read_csv(path_or_buf)
        # TODO: common_df.read_csv sets the index of the dataframe to be fips, date, however
        # most of the calling code expects fips and date to not be in an index.
        # In the future, it would be good to standardize around index fields.
        df = df.reset_index()
        return cls(df)


def _add_location_id(df: pd.DataFrame):
    """Adds the location_id column derived from FIPS, inplace."""
    if CommonFields.LOCATION_ID in df.columns:
        raise ValueError("location_id already in DataFrame")
    df[CommonFields.LOCATION_ID] = df[CommonFields.FIPS].apply(pipeline.fips_to_location_id)


def _add_fips_if_missing(df: pd.DataFrame):
    """Adds the FIPS column derived from location_id, inplace."""
    if CommonFields.FIPS not in df.columns:
        df[CommonFields.FIPS] = df[CommonFields.LOCATION_ID].apply(pipeline.location_id_to_fips)


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
        _log.warning(f"Re-adding empty columns: {missing_columns}")
        wide = wide.reindex(columns=[*wide.columns, *missing_columns])
    # Make all non-GEO_DATA_COLUMNS numeric so that aggregation functions work on them.
    numeric_columns = list(all_columns - set(GEO_DATA_COLUMNS))
    wide[numeric_columns] = wide[numeric_columns].apply(pd.to_numeric, axis=0)

    assert wide.index.names == [CommonFields.LOCATION_ID]

    return wide


# An empty pd.Series with the structure expected for the provenance attribute. Use this when a dataset
# does not have provenance information.
_EMPTY_PROVENANCE_SERIES = pd.Series(
    [],
    name=PdFields.PROVENANCE,
    dtype="str",
    index=pd.MultiIndex.from_tuples([], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE]),
)

# An empty pd.DataFrame with the structure expected for the static attribute. Use this when
# a dataset does not have any static values.
_EMPTY_REGIONAL_ATTRIBUTES_DF = pd.DataFrame([], index=pd.Index([], name=CommonFields.LOCATION_ID))


_EMPTY_TIMESERIES_WIDE_DATES_DF = pd.DataFrame(
    [],
    dtype=float,
    index=pd.MultiIndex.from_tuples([], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE]),
    columns=pd.Index([], name=CommonFields.DATE),
)


@final
@dataclass(frozen=True, eq=False)  # Instances are large so compare by id instead of value
class MultiRegionDataset(SaveableDatasetInterface):
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

    # `provenance` is an array of str with a MultiIndex with names LOCATION_ID and VARIABLE.
    provenance: pd.Series = _EMPTY_PROVENANCE_SERIES

    # `data` has a simple integer index and columns from CommonFields. DATE and LOCATION_ID must
    # be non-null in every row.
    @cached_property
    def data(self) -> pd.DataFrame:
        # TODO(tom): Remove this attribute. There are only a few users of it.
        df = self._geo_data.join(self.timeseries).drop(columns=[CommonFields.FIPS], errors="ignore")
        return df.reset_index()

    @cached_property
    def _geo_data(self) -> pd.DataFrame:
        return self.static.loc[:, self.static.columns.isin(GEO_DATA_COLUMNS)]

    @cached_property
    def dataset_type(self) -> DatasetType:
        return DatasetType.MULTI_REGION

    @cached_property
    def data_with_fips(self) -> pd.DataFrame:
        """data with FIPS column, use `data` when FIPS is not need."""
        data_copy = self.data.copy()
        _add_fips_if_missing(data_copy)
        return data_copy

    @cached_property
    def static_data_with_fips(self) -> pd.DataFrame:
        """_latest_data with FIPS column and LOCATION_ID index.

        TODO(tom): This data is usually accessed via OneRegionTimeseriesDataset. Retire this
        property.
        """
        data_copy = self.static.reset_index()
        _add_fips_if_missing(data_copy)
        return data_copy.set_index(CommonFields.LOCATION_ID)

    @lru_cache(maxsize=None)
    def _static_and_timeseries_latest_with_fips(self) -> pd.DataFrame:
        """Static values merged with the latest timeseries values."""
        return _merge_attributes(
            self._timeseries_latest_values().reset_index(), self.static.reset_index()
        )

    @classmethod
    def load_csv(cls, path_or_buf: Union[pathlib.Path, TextIO]):
        return MultiRegionDataset.from_csv(path_or_buf)

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

    def timeseries_wide_dates(self) -> pd.DataFrame:
        """Returns the timeseries in a DataFrame with LOCATION_ID, VARIABLE index and DATE columns."""
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

    @staticmethod
    def from_timeseries_wide_dates_df(timeseries_wide_dates: pd.DataFrame) -> "MultiRegionDataset":
        """Make a new dataset from a DataFrame as returned by timeseries_wide_dates."""
        assert timeseries_wide_dates.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
        assert timeseries_wide_dates.columns.names == [CommonFields.DATE]
        timeseries_wide_dates.columns: pd.DatetimeIndex = pd.to_datetime(
            timeseries_wide_dates.columns
        )
        timeseries_wide_variables = timeseries_wide_dates.stack().unstack(PdFields.VARIABLE)
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
            set(TimeseriesDataset.INDEX_FIELDS) | set(GEO_DATA_COLUMNS)
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
        return self.add_provenance_series(
            pd.Series([], dtype=str, name=PdFields.PROVENANCE).reindex(
                self.timeseries_wide_dates().index, fill_value=provenance
            )
        )

    def add_provenance_series(self, provenance: pd.Series) -> "MultiRegionDataset":
        """Returns a new object containing data in self and given provenance information."""
        if not self.provenance.empty:
            raise NotImplementedError("TODO(tom): add support for merging provenance data")

        # Make a sorted series. The order doesn't matter and sorting makes the order depend only on
        # what is represented, not the order it appears in the input.
        return dataclasses.replace(self, provenance=provenance.sort_index())

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
    def from_timeseries_and_latest(
        ts: TimeseriesDataset, latest: LatestValuesDataset
    ) -> "MultiRegionDataset":
        """Converts legacy FIPS to new LOCATION_ID and calls `from_timeseries_df` to finish construction."""
        timeseries_df = ts.data.copy()
        _add_location_id(timeseries_df)
        dataset = MultiRegionDataset.from_geodata_timeseries_df(timeseries_df)

        latest_df = latest.data.copy()
        _add_location_id(latest_df)
        dataset = dataset.add_static_values(latest_df)

        if ts.provenance is not None:
            # Check that current index is as expected. Names will be fixed after remapping, below.
            assert ts.provenance.index.names == [CommonFields.FIPS, PdFields.VARIABLE]
            provenance = ts.provenance.copy()
            provenance.index = provenance.index.map(
                lambda i: (pipeline.fips_to_location_id(i[0]), i[1])
            )
            provenance.index.rename([CommonFields.LOCATION_ID, PdFields.VARIABLE], inplace=True)
            provenance.rename(PdFields.PROVENANCE, inplace=True)
            dataset = dataset.add_provenance_series(provenance)

        # TODO(tom): Either copy latest.provenance to its own series, blend with the timeseries
        # provenance (though some variable names are the same), or retire latest as a separate thing upstream.

        return dataset

    @staticmethod
    def from_latest(latest: LatestValuesDataset) -> "MultiRegionDataset":
        """Creates a new MultiRegionDataset with static data from latest and no timeseries data."""
        return MultiRegionDataset.from_timeseries_and_latest(
            TimeseriesDataset(pd.DataFrame([], columns=[CommonFields.FIPS, CommonFields.DATE])),
            latest,
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
        assert isinstance(self.provenance, pd.Series)
        assert self.provenance.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
        assert self.provenance.index.is_unique
        assert self.provenance.index.is_monotonic_increasing
        assert self.provenance.name == PdFields.PROVENANCE
        # Check that all provenance location_id are in timeseries location_id
        assert (
            self.provenance.index.get_level_values(CommonFields.LOCATION_ID)
            .difference(self.timeseries.index.get_level_values(CommonFields.LOCATION_ID))
            .empty
        )
        self._check_fips()

    def _check_fips(self):
        """Logs a message if FIPS is present for a subset of regions, a bit of a mysterious problem."""
        if CommonFields.FIPS in self.static:
            missing_fips = self.static[CommonFields.FIPS].isna()
            if missing_fips.any():
                columns = self.static.columns.intersection(GEO_DATA_COLUMNS)
                _log.info(
                    f"Missing fips for some regions:\n{self.static.loc[missing_fips, columns]}"
                )

    def append_regions(self, other: "MultiRegionDataset") -> "MultiRegionDataset":
        common_location_id = self.static.index.intersection(other.static.index)
        if not common_location_id.empty:
            raise ValueError("Do not use append_regions with duplicate location_id")
        timeseries_df = pd.concat([self.timeseries, other.timeseries]).sort_index()
        static_df = pd.concat([self.static, other.static]).sort_index()
        provenance = pd.concat([self.provenance, other.provenance]).sort_index()
        return MultiRegionDataset(
            timeseries=timeseries_df, static=static_df, provenance=provenance,
        )

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
        provenance_dict = self._location_id_provenance_dict(region.location_id)

        return OneRegionTimeseriesDataset(
            region=region, data=ts_df, latest=latest_dict, provenance=provenance_dict,
        )

    def _location_id_provenance_dict(self, location_id: str) -> dict:
        """Returns the provenance dict of a location_id."""
        try:
            provenance_series = self.provenance.loc[location_id]
        except KeyError:
            return {}
        else:
            return provenance_series[provenance_series.notna()].to_dict()

    def _location_id_latest_dict(self, location_id: str) -> dict:
        """Returns the latest values dict of a location_id."""
        try:
            attributes_series = self._static_and_timeseries_latest_with_fips().loc[location_id, :]
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
        provenance_mask = self.provenance.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        provenance = self.provenance.loc[provenance_mask, :]
        return MultiRegionDataset(
            timeseries=timeseries_df, static=static_df, provenance=provenance,
        )

    def get_subset(
        self,
        aggregation_level: Optional[AggregationLevel] = None,
        fips: Optional[str] = None,
        state: Optional[str] = None,
        states: Optional[List[str]] = None,
        exclude_county_999: bool = False,
    ) -> "MultiRegionDataset":
        """Returns a new object containing data for a subset of the regions in `self`."""
        rows_key = dataset_utils.make_rows_key(
            self.static,
            aggregation_level=aggregation_level,
            fips=fips,
            state=state,
            states=states,
            exclude_county_999=exclude_county_999,
        )
        location_ids = self.static.loc[rows_key, :].index
        return self.get_locations_subset(location_ids)

    def get_counties(self, after: Optional[datetime.datetime] = None) -> "MultiRegionDataset":
        return self.get_subset(aggregation_level=AggregationLevel.COUNTY)._trim_timeseries(
            after=after
        )

    def _trim_timeseries(self, *, after: datetime.datetime) -> "MultiRegionDataset":
        """Returns a new object containing only timeseries data after given date."""
        ts_rows_mask = self.timeseries.index.get_level_values(CommonFields.DATE) > after
        return dataclasses.replace(self, timeseries=self.timeseries.loc[ts_rows_mask, :])

    def groupby_region(self) -> pandas.core.groupby.generic.DataFrameGroupBy:
        return self.timeseries.groupby(CommonFields.LOCATION_ID)

    def to_timeseries(self) -> TimeseriesDataset:
        """Returns a `TimeseriesDataset` of this data.

        This method exists for interim use when calling code that doesn't use MultiRegionDataset.
        """
        return TimeseriesDataset(
            self.data_with_fips, provenance=None if self.provenance.empty else self.provenance
        )

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
        )
        # Only keep provenance information for timeseries in the new timeseries_wide_dates.
        provenance = self.provenance.reindex(
            self.provenance.index.intersection(timeseries_wide_dates.index)
        ).sort_index()
        return dataclasses.replace(
            self, timeseries=timeseries_wide_variables, provenance=provenance
        )

    def to_csv(self, path: pathlib.Path, write_timeseries_latest_values=False):
        """Persists timeseries to CSV.

        Args:
            path: Path to write to.
            write_timeseries_latest_values: write both explicit set static attributes and timeseries
              values derived from timeseries. Mostly exists to compare to old files created when latest
              values were calculated upstream.
        """
        if write_timeseries_latest_values:
            latest_data = self._static_and_timeseries_latest_with_fips().reset_index()
        else:
            latest_data = self.static_data_with_fips.reset_index()
        # A DataFrame with timeseries data and latest data (with DATE=NaT) together
        combined = pd.concat([self.data_with_fips, latest_data], ignore_index=True)
        assert combined[CommonFields.LOCATION_ID].notna().all()
        common_df.write_csv(
            combined, path, structlog.get_logger(), [CommonFields.LOCATION_ID, CommonFields.DATE]
        )
        if not self.provenance.empty:
            provenance_path = str(path).replace(".csv", "-provenance.csv")
            self.provenance.sort_index().rename(PdFields.PROVENANCE).to_csv(provenance_path)

    def drop_column_if_present(self, column: str) -> "MultiRegionDataset":
        """Drops the specified column from the timeseries if it exists"""
        timeseries_df = self.timeseries.drop(column, axis="columns", errors="ignore")
        static_df = self.static.drop(column, axis="columns", errors="ignore")
        provenance = self.provenance[
            self.provenance.index.get_level_values(PdFields.VARIABLE) != column
        ]
        return MultiRegionDataset(
            timeseries=timeseries_df, static=static_df, provenance=provenance,
        )

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
        combined_provenance = pd.concat([self.provenance, other.provenance]).sort_index()
        return MultiRegionDataset(
            timeseries=combined_df, static=self.static, provenance=combined_provenance,
        )

    def iter_one_regions(self) -> Iterable[Tuple[Region, OneRegionTimeseriesDataset]]:
        """Iterates through all the regions in this object"""
        for location_id, timeseries_group in self.timeseries.groupby(
            CommonFields.LOCATION_ID, as_index=True
        ):
            latest_dict = self._location_id_latest_dict(location_id)
            provenance_dict = self._location_id_provenance_dict(location_id)
            region = Region.from_location_id(location_id)
            yield region, OneRegionTimeseriesDataset(
                region, timeseries_group.reset_index(), latest_dict, provenance=provenance_dict,
            )

    def get_county_name(self, *, region: pipeline.Region) -> str:
        return self.static.at[region.location_id, CommonFields.COUNTY]


def _remove_padded_nans(df, columns):
    if df[columns].isna().all(axis=None):
        return df.loc[[False] * len(df), :].reset_index(drop=True)

    first_valid_index = min(df[column].first_valid_index() for column in columns)
    last_valid_index = max(df[column].last_valid_index() for column in columns)
    df = df.iloc[first_valid_index : last_valid_index + 1]
    return df.reset_index(drop=True)


def _diff_preserving_first_value(series):

    series_diff = series.diff()
    first_valid_index = series.first_valid_index()
    if first_valid_index is None:
        return series_diff

    series_diff[first_valid_index] = series[series.first_valid_index()]
    return series_diff


def add_new_cases(dataset_in: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""
    df_copy = dataset_in.timeseries.copy()
    grouped_df = dataset_in.groupby_region()
    # Calculating new cases using diff will remove the first detected value from the case series.
    # We want to capture the first day a region reports a case. Since our data sources have
    # been capturing cases in all states from the beginning of the pandemic, we are treating
    # The first days as appropriate new case data.
    new_cases = grouped_df[CommonFields.CASES].apply(_diff_preserving_first_value)

    # Remove the occasional negative case adjustments.
    new_cases[new_cases < 0] = pd.NA

    df_copy[CommonFields.NEW_CASES] = new_cases

    new_timeseries = dataclasses.replace(dataset_in, timeseries=df_copy)

    return new_timeseries


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
    df_copy.loc[to_exclude, CommonFields.NEW_CASES] = None

    new_timeseries = dataclasses.replace(timeseries, timeseries=df_copy)

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


# Column for the aggregated location_id
LOCATION_ID_AGG = "location_id_agg"


def _aggregate_dataframe_by_region(
    df_in: pd.DataFrame, location_id_map: Mapping[str, str]
) -> pd.DataFrame:
    """Aggregates a DataFrame using given region map. The output contains dates iff the input does."""
    df = df_in.copy()  # Copy because the index is modified below

    if CommonFields.DATE in df.index.names:
        groupby_columns = [LOCATION_ID_AGG, CommonFields.DATE, PdFields.VARIABLE]
    else:
        groupby_columns = [LOCATION_ID_AGG, PdFields.VARIABLE]

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

    # Make a Series with index `location_id_agg` containing the count of regions to be
    # aggregated for that region.
    location_id_agg_count = (
        pd.Series(list(location_id_map.values()))
        .value_counts()
        .rename_axis(LOCATION_ID_AGG)
        .rename("location_id_agg_count")
    )

    # Aggregate by location_id_agg, optional date and variable. Keep the sum and count of
    # input values.
    long_agg = long_all_values.groupby(groupby_columns, sort=False).agg(["sum", "count"])
    # Join the count of regions in each location_id_agg
    long_agg = long_agg.join(location_id_agg_count, on=LOCATION_ID_AGG)

    # Only keep aggregated values where the count of aggregated values is the same as the
    # count of input regions.
    long_agg_all_present = long_agg.loc[long_agg["count"] == long_agg["location_id_agg_count"]]

    df_out = (
        long_agg_all_present["sum"]
        .unstack()
        .rename_axis(index={LOCATION_ID_AGG: CommonFields.LOCATION_ID})
        .sort_index()
        .reindex(columns=df_in.columns)
    )
    assert df_in.index.names == df_out.index.names
    return df_out


def aggregate_regions(
    dataset_in: MultiRegionDataset,
    aggregate_map: Mapping[Region, Region],
    aggregate_level: AggregationLevel,
) -> MultiRegionDataset:
    dataset_in = dataset_in.get_regions_subset(aggregate_map.keys())
    location_id_map = {
        region_in.location_id: region_agg.location_id
        for region_in, region_agg in aggregate_map.items()
    }
    timeseries_out = _aggregate_dataframe_by_region(dataset_in.timeseries, location_id_map)

    # TODO(tom): Do something smarter with non-number columns in static. Currently they are
    # silently dropped.
    static_out = _aggregate_dataframe_by_region(
        dataset_in.static.select_dtypes(include="number"), location_id_map
    )
    static_out[CommonFields.AGGREGATE_LEVEL] = aggregate_level.value

    return MultiRegionDataset(timeseries=timeseries_out, static=static_out)


class DatasetName(str):
    """Human readable name for a dataset, providing some type safety."""

    pass


def _to_datasets_wide_dates_map(
    datasets: Mapping[DatasetName, MultiRegionDataset]
) -> Mapping[DatasetName, pd.DataFrame]:
    datasets_wide = {name: ds.timeseries_wide_dates() for name, ds in datasets.items()}
    # Find the earliest and latest dates to make a range covering all timeseries.
    dates = pd.DatetimeIndex(
        np.hstack(
            list(df.columns.get_level_values(CommonFields.DATE) for df in datasets_wide.values())
        )
    )
    start_date = dates.min()
    end_date = dates.max()
    input_date_range = pd.date_range(start=start_date, end=end_date, name=CommonFields.DATE)
    datasets_wide_reindexed = {
        name: df.reorder_levels([PdFields.VARIABLE, CommonFields.LOCATION_ID]).reindex(
            columns=input_date_range
        )
        for name, df in datasets_wide.items()
    }
    return datasets_wide_reindexed


def combined_datasets(
    datasets: Mapping[DatasetName, MultiRegionDataset],
    field_dataset_source: Mapping[FieldName, List[DatasetName]],
) -> MultiRegionDataset:
    relevant_columns = list(field_dataset_source.keys())

    datasets_wide = _to_datasets_wide_dates_map(datasets)
    field_values_map = {}
    provenance_to_concat = []
    for field, datasets_list in field_dataset_source.items():
        location_id_so_far = None
        values_to_concat = []
        for dataset_name in datasets_list:
            field_wide_df = datasets_wide[dataset_name].loc[field, :]
            assert field_wide_df.index.names == [CommonFields.LOCATION_ID]
            if location_id_so_far is None:
                values_to_concat.append(field_wide_df)
                location_id_so_far = field_wide_df.index
                provenance_to_concat.append(
                    datasets[dataset_name].provenance.loc[slice(None), slice(field)]
                )
            else:
                selected_location_id = field_wide_df.index.difference(location_id_so_far)
                values_to_concat.append(field_wide_df.loc[selected_location_id, :])
                location_id_so_far = location_id_so_far.union(selected_location_id).sort_values()
                provenance_to_concat.append(
                    datasets[dataset_name].provenance.loc[selected_location_id, slice(field)]
                )
        field_values_map[field] = pd.concat(values_to_concat)
    output_wide = pd.concat(field_values_map, names=[PdFields.VARIABLE, CommonFields.LOCATION_ID])
    output_provenance = pd.concat(provenance_to_concat, verify_integrity=True).sort_index()
    assert output_wide.index.names == [PdFields.VARIABLE, CommonFields.LOCATION_ID]
    assert output_wide.columns.names == [CommonFields.DATE]
    return MultiRegionDataset(
        timeseries=output_wide.stack().unstack(PdFields.VARIABLE).sort_index(),
        provenance=output_provenance,
    )
