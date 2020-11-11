import dataclasses
import datetime
import pathlib
from dataclasses import dataclass
from typing import Any
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List, Optional, Union, TextIO
from typing import Sequence
from typing import Tuple

from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import PdFields
from typing_extensions import final

import pandas as pd
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

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.TIMESERIES

    @property
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


def _geodata_df_to_regional_attributes_df(geodata_df: pd.DataFrame) -> pd.DataFrame:
    assert geodata_df.index.names == [None]  # [CommonFields.LOCATION_ID, CommonFields.DATE]
    deduped_values = geodata_df.drop_duplicates().set_index(CommonFields.LOCATION_ID)
    duplicates = deduped_values.index.duplicated(keep=False)
    if duplicates.any():
        _log.warning("Conflicting geo data", duplicates=deduped_values.loc[duplicates, :])
        deduped_values = deduped_values.loc[~deduped_values.index.duplicated(keep="first"), :]
    return deduped_values.sort_index()


def _merge_attributes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merges the values in two DataFrame objects, with not-NA values in df2 merged into df1."""
    assert df1.index.names == [None]
    assert df2.index.names == [None]

    # Get the union of all location_id and columns in the input
    all_locations = sorted(set(df1[CommonFields.LOCATION_ID]) | set(df2[CommonFields.LOCATION_ID]))
    all_columns = set(df1.columns.union(df2.columns)) - {CommonFields.LOCATION_ID}
    # Transform from a column for each metric to a row for every value. Put df2 first so
    # the duplicate dropping keeps it.
    long = pd.melt(pd.concat([df2, df1]), id_vars=[CommonFields.LOCATION_ID])
    # Drop duplicate values for the same LOCATION_ID, VARIABLE
    long_deduped = long.drop_duplicates()
    long_deduped = long_deduped.set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE])[
        PdFields.VALUE
    ].dropna()
    # If the LOCATION_ID, VARIABLE index contains duplicates then df2 is changing a value.
    # This is not expected so log a warning before dropping the old value.
    dups = long_deduped.index.duplicated(keep=False)
    if dups.any():
        _log.warning(f"Duplicates:\n{long_deduped.loc[dups, :]}")
        long_deduped = long_deduped.loc[long_deduped.index.duplicated(keep="first"), :]
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


# An empty pd.Series with the structure expected for the provenance series. Use this when a dataset
# does not have provenance information. Make a copy for an extra level of protection against
# unrelated objects sharing common objects.
_EMPTY_PROVENANCE_SERIES = pd.Series(
    [],
    name=PdFields.PROVENANCE,
    dtype="str",
    index=pd.MultiIndex.from_tuples([], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE]),
)

_EMPTY_REGIONAL_ATTRIBUTES_DF = pd.DataFrame([], index=pd.Index([], name=CommonFields.LOCATION_ID))


@final
@dataclass(frozen=True)
class MultiRegionDataset(SaveableDatasetInterface):
    """A set of timeseries and constant values from any number of regions.

    Methods named `append_...` return a new object with more regions of data. Methods named `add_...` and
    `join_...` return a new object with more data about the same regions, such as new metrics and provenance
    information.
    """

    _timeseries: pd.DataFrame

    _regional_attributes: pd.DataFrame = _EMPTY_REGIONAL_ATTRIBUTES_DF

    # `provenance` is an array of str with a MultiIndex with names LOCATION_ID and VARIABLE.
    _provenance: pd.Series = _EMPTY_PROVENANCE_SERIES

    # `data` contains columns from CommonFields and simple integer index. DATE and LOCATION_ID must
    # be non-null in every row.
    @property
    def data(self) -> pd.DataFrame:
        cc = self._regional_geo_data.join(self._timeseries)
        # cc = pd.concat([self._regional_geo_data, self._timeseries], axis=1)
        return cc.reset_index()

    @property
    def _regional_geo_data(self) -> pd.DataFrame:
        return self._regional_attributes.loc[
            :, self._regional_attributes.columns.isin(GEO_DATA_COLUMNS)
        ]

    @property
    def _regional_non_geo_data(self) -> pd.DataFrame:
        return self._regional_attributes.loc[
            :, ~self._regional_attributes.columns.isin(GEO_DATA_COLUMNS)
        ]

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.MULTI_REGION

    @property
    def data_with_fips(self) -> pd.DataFrame:
        """data with FIPS column, use `data` when FIPS is not need."""
        data_copy = self.data.copy()
        _add_fips_if_missing(data_copy)
        return data_copy

    @property
    def latest_data_with_fips(self) -> pd.DataFrame:
        """_latest_data with FIPS column and LOCATION_ID index.

        TODO(tom): This data is usually accessed via OneRegionTimeseriesDataset. Retire this
        property.
        """
        data_copy = self._regional_attributes.reset_index()
        _add_fips_if_missing(data_copy)
        return data_copy.set_index(CommonFields.LOCATION_ID)

    @classmethod
    def load_csv(cls, path_or_buf: Union[pathlib.Path, TextIO]):
        return MultiRegionDataset.from_csv(path_or_buf)

    def timeseries_long(self, columns: List[common_fields.FieldName]) -> pd.DataFrame:
        """Returns a subset of the data in a long format DataFrame, where all values are in a single column.

        Returns: a DataFrame with columns LOCATION_ID, DATE, VARIABLE, VALUE
        """

        key_cols = [CommonFields.LOCATION_ID, CommonFields.DATE]
        long = (
            self.data.loc[:, key_cols + columns]
            .melt(id_vars=key_cols, value_vars=columns)
            .dropna()
            .reset_index(drop=True)
        )
        long[PdFields.VALUE].apply(pd.to_numeric)
        return long

    def _timeseries_latest_values(self) -> pd.DataFrame:
        """Returns the latest value for every region and metric, derived from _timeseries."""
        # _timeseries is already sorted by DATE with the latest at the bottom.
        long = self._timeseries.stack().droplevel(CommonFields.DATE)
        # `long` has MultiIndex with LOCATION_ID and VARIABLE (added by stack). Keep only the last
        # row with each index to get the last value for each date.
        last_mask = ~long.index.duplicated(keep="last")
        return long.loc[last_mask, :].unstack()

    @staticmethod
    def from_timeseries_df(timeseries_df: pd.DataFrame) -> "MultiRegionDataset":
        """Make a new dataset from a DataFrame containing timeseries (real-valued metrics) and
        geo data (county name etc)."""
        assert timeseries_df.index.names == [None]
        if CommonFields.FIPS in timeseries_df.columns:
            timeseries_df = timeseries_df.drop(columns=[CommonFields.FIPS])
        timeseries_df = timeseries_df.set_index([CommonFields.LOCATION_ID, CommonFields.DATE])

        geodata_column_mask = timeseries_df.columns.isin(
            set(TimeseriesDataset.INDEX_FIELDS) | set(GEO_DATA_COLUMNS)
        )
        regional_attribute_df = _geodata_df_to_regional_attributes_df(
            timeseries_df.loc[:, geodata_column_mask]
            .reset_index()
            .drop(columns=[CommonFields.DATE])
        )

        timeseries_df = timeseries_df.loc[:, ~geodata_column_mask].sort_index()

        return MultiRegionDataset(
            _timeseries=timeseries_df, _regional_attributes=regional_attribute_df
        )

    def add_latest_df(self, latest_df: pd.DataFrame) -> "MultiRegionDataset":
        """Returns a new object with not NA values in `latest_df` added as regional attributes."""
        combined_attributes = _merge_attributes(self._regional_attributes.reset_index(), latest_df)
        assert combined_attributes.index.names == [CommonFields.LOCATION_ID]
        return dataclasses.replace(self, _regional_attributes=combined_attributes)

    def add_provenance_csv(self, path_or_buf: Union[pathlib.Path, TextIO]) -> "MultiRegionDataset":
        df = pd.read_csv(path_or_buf)
        if PdFields.VALUE in df.columns:
            # Handle older CSV files that used 'value' header for provenance.
            df = df.rename(columns={PdFields.VALUE: PdFields.PROVENANCE})
        series = df.set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE])[PdFields.PROVENANCE]
        return self.add_provenance_series(series)

    def add_provenance_series(self, provenance: pd.Series) -> "MultiRegionDataset":
        """Returns a new object containing data in self and given provenance information."""
        if not self._provenance.empty:
            raise NotImplementedError("TODO(tom): add support for merging provenance data")

        # Make a sorted series. The order doesn't matter and sorting makes the order depend only on
        # what is represented, not the order it appears in the input.
        return dataclasses.replace(self, _provenance=provenance.sort_index())

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

        dataset = MultiRegionDataset.from_timeseries_df(timeseries_df)
        if not latest_df.empty:
            dataset = dataset.add_latest_df(latest_df.drop(columns=[CommonFields.DATE]))

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
        dataset = MultiRegionDataset.from_timeseries_df(timeseries_df)

        latest_df = latest.data.copy()
        _add_location_id(latest_df)
        dataset = dataset.add_latest_df(latest_df)

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

    def __post_init__(self):
        """Checks that attributes of this object meet certain expectations."""
        # These asserts provide runtime-checking and a single place for humans reading the code to
        # check what is expected of the attributes, beyond type.
        # _timeseries.index order is important for _timeseries_latest_values correctness.
        assert self._timeseries.index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
        assert self._timeseries.index.is_unique
        assert self._timeseries.index.is_monotonic_increasing
        assert self._regional_attributes.index.names == [CommonFields.LOCATION_ID]
        assert self._regional_attributes.index.is_unique
        assert self._regional_attributes.index.is_monotonic_increasing
        assert isinstance(self._provenance, pd.Series)
        assert self._provenance.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
        assert self._provenance.index.is_unique
        assert self._provenance.index.is_monotonic_increasing
        assert self._provenance.name == PdFields.PROVENANCE

    def append_regions(self, other: "MultiRegionDataset") -> "MultiRegionDataset":
        timeseries_df = pd.concat([self._timeseries, other._timeseries]).sort_index()
        regional_attributes = pd.concat(
            [self._regional_attributes, other._regional_attributes]
        ).sort_index()
        provenance = pd.concat([self._provenance, other._provenance]).sort_index()
        return MultiRegionDataset(
            _timeseries=timeseries_df,
            _regional_attributes=regional_attributes,
            _provenance=provenance,
        )

    def get_one_region(self, region: Region) -> OneRegionTimeseriesDataset:
        ts_df = self.data.loc[self.data[CommonFields.LOCATION_ID] == region.location_id, :]
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
            provenance_series = self._provenance.loc[location_id]
        except KeyError:
            return {}
        else:
            return provenance_series[provenance_series.notna()].to_dict()

    def _location_id_latest_dict(self, location_id: str) -> dict:
        """Returns the latest values dict of a location_id."""
        try:
            attributes_series = self._regional_attributes.loc[location_id, :]
        except KeyError:
            attributes_series = pd.Series([], dtype=object)
        # Split attributes_series into a dict with NA values and dict of not-NA values
        attributes_none_dict = {
            key: None for key in attributes_series.loc[attributes_series.isna()].index
        }
        attributes_notna_dict = attributes_series.dropna().to_dict()

        try:
            timeseries_latest_series = self._timeseries_latest_values().loc[location_id]
        except KeyError:
            timeseries_latest_series = pd.Series([], dtype=object)

        # Start with NA values from attributes_series
        result_dict = attributes_none_dict
        # override them with latest values from _timeseries_latest_values
        result_dict.update(timeseries_latest_series.dropna().to_dict())
        # and override them with not-NA value from attributes_series
        result_dict.update(attributes_notna_dict)
        return result_dict

    def get_regions_subset(self, regions: Collection[Region]) -> "MultiRegionDataset":
        location_ids = pd.Index(sorted(r.location_id for r in regions))
        return self.get_locations_subset(location_ids)

    def get_locations_subset(self, location_ids: Collection[str]) -> "MultiRegionDataset":
        timeseries_mask = self._timeseries.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        timeseries_df = self._timeseries.loc[timeseries_mask, :]
        regional_attributes_mask = self._regional_attributes.index.get_level_values(
            CommonFields.LOCATION_ID
        ).isin(location_ids)
        regional_attributes = self._regional_attributes.loc[regional_attributes_mask, :]
        provenance_mask = self._provenance.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        provenance = self._provenance.loc[provenance_mask, :]
        return MultiRegionDataset(
            _timeseries=timeseries_df,
            _regional_attributes=regional_attributes,
            _provenance=provenance,
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
            self._regional_attributes,
            aggregation_level=aggregation_level,
            fips=fips,
            state=state,
            states=states,
            exclude_county_999=exclude_county_999,
        )
        location_ids = self._regional_attributes.loc[rows_key, :].index
        return self.get_locations_subset(location_ids)

    def get_counties(self, after: Optional[datetime.datetime] = None) -> "MultiRegionDataset":
        return self.get_subset(aggregation_level=AggregationLevel.COUNTY)._trim_timeseries(
            after=after
        )

    def _trim_timeseries(self, *, after: datetime.datetime) -> "MultiRegionDataset":
        """Returns a new object containing only timeseries data after given date."""
        ts_rows_mask = self._timeseries.index.get_level_values(CommonFields.DATE) > after
        # `after` doesn't make sense for latest because it doesn't contain date information.
        # Keep latest data for regions that are in the new timeseries DataFrame.
        # TODO(tom): Replace this with re-calculating latest data from the new timeseries so that
        # metrics which no longer have real values are excluded.
        return dataclasses.replace(self, _timeseries=self._timeseries.loc[ts_rows_mask, :])

    def groupby_region(self) -> pandas.core.groupby.generic.DataFrameGroupBy:
        return self._timeseries.groupby(CommonFields.LOCATION_ID)

    @property
    def empty(self) -> bool:
        return self.data.empty

    def to_timeseries(self) -> TimeseriesDataset:
        """Returns a `TimeseriesDataset` of this data.

        This method exists for interim use when calling code that doesn't use MultiRegionDataset.
        """
        return TimeseriesDataset(
            self.data_with_fips, provenance=None if self._provenance.empty else self._provenance
        )

    def to_csv(self, path: pathlib.Path):
        """Persists timeseries to CSV.

        Args:
            path: Path to write to.
        """
        # A DataFrame with timeseries data and latest data (with DATE=NaT) together
        combined = pd.concat(
            [self.data_with_fips, self.latest_data_with_fips.reset_index()], ignore_index=True
        )
        assert combined[CommonFields.LOCATION_ID].notna().all()
        common_df.write_csv(
            combined, path, structlog.get_logger(), [CommonFields.LOCATION_ID, CommonFields.DATE]
        )
        if not self._provenance.empty:
            provenance_path = str(path).replace(".csv", "-provenance.csv")
            self._provenance.sort_index().rename(PdFields.PROVENANCE).to_csv(provenance_path)

    def drop_column_if_present(self, column: str) -> "MultiRegionDataset":
        """Drops the specified column from the timeseries if it exists"""
        timeseries_df = self._timeseries.drop(column, axis="columns", errors="ignore")
        regional_attributes = self._regional_attributes.drop(
            column, axis="columns", errors="ignore"
        )
        provenance = self._provenance[
            self._provenance.index.get_level_values(PdFields.VARIABLE) != column
        ]
        return MultiRegionDataset(
            _timeseries=timeseries_df,
            _regional_attributes=regional_attributes,
            _provenance=provenance,
        )

    def join_columns(self, other: "MultiRegionDataset") -> "MultiRegionDataset":
        """Joins the timeseries columns in `other` with those in `self`.

        Args:
            other: The timeseries dataset to join with `self`. all columns except the "geo" columns
                   will be joined into `self`.
        """
        other_non_geo_attributes = set(other._regional_attributes.columns) - set(GEO_DATA_COLUMNS)
        if other_non_geo_attributes:
            raise NotImplementedError(
                f"join with other with attributes {other_non_geo_attributes} not supported"
            )
        common_ts_columns = set(other._timeseries.columns) & set(self._timeseries.columns)
        if common_ts_columns:
            # columns to be joined need to be disjoint
            raise ValueError(f"Columns are in both dataset: {common_ts_columns}")
        # TODO(tom): fix geo columns check, no later than when self.data is changed to contain only
        # timeseries
        # self_common_geo_columns = self_df.loc[:, common_geo_columns].fillna("")
        # other_common_geo_columns = other_df.loc[:, common_geo_columns].fillna("")
        # try:
        #    if (self_common_geo_columns != other_common_geo_columns).any(axis=None):
        #        unequal_rows = (self_common_geo_columns != other_common_geo_columns).any(axis=1)
        #        _log.info(
        #            "Geo data unexpectedly varies",
        #            self_rows=self_df.loc[unequal_rows, common_geo_columns],
        #            other_rows=other_df.loc[unequal_rows, common_geo_columns],
        #        )
        #        raise ValueError("Geo data unexpectedly varies")
        # except Exception:
        #    _log.exception(f"Comparing df {self_common_geo_columns} to {other_common_geo_columns}")
        #    raise
        combined_df = pd.concat([self._timeseries, other._timeseries], axis=1)
        combined_provenance = pd.concat([self._provenance, other._provenance]).sort_index()
        return MultiRegionDataset(
            _timeseries=combined_df,
            _regional_attributes=self._regional_attributes,
            _provenance=combined_provenance,
        )

    def iter_one_regions(self) -> Iterable[Tuple[Region, OneRegionTimeseriesDataset]]:
        """Iterates through all the regions in this object"""
        for location_id, data_group in self.data.groupby(CommonFields.LOCATION_ID):
            latest_dict = self._location_id_latest_dict(location_id)
            provenance_dict = self._location_id_provenance_dict(location_id)
            region = Region.from_location_id(location_id)
            yield region, OneRegionTimeseriesDataset(
                region, data_group, latest_dict, provenance=provenance_dict,
            )


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


def _add_new_cases_to_latest(timeseries_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
    assert latest_df.index.names == [CommonFields.LOCATION_ID]
    latest_new_cases = dataset_utils.build_latest_for_column(timeseries_df, CommonFields.NEW_CASES)

    latest_copy = latest_df.copy()
    latest_copy[CommonFields.NEW_CASES] = latest_new_cases
    return latest_copy


def add_new_cases(timeseries: MultiRegionDataset) -> MultiRegionDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""
    df_copy = timeseries._timeseries.copy()
    grouped_df = timeseries.groupby_region()
    # Calculating new cases using diff will remove the first detected value from the case series.
    # We want to capture the first day a region reports a case. Since our data sources have
    # been capturing cases in all states from the beginning of the pandemic, we are treating
    # The first days as appropriate new case data.
    new_cases = grouped_df[CommonFields.CASES].apply(_diff_preserving_first_value)

    # Remove the occasional negative case adjustments.
    new_cases[new_cases < 0] = pd.NA

    df_copy[CommonFields.NEW_CASES] = new_cases
    regional_attributes = _add_new_cases_to_latest(df_copy, timeseries._regional_attributes)

    new_timeseries = MultiRegionDataset(
        _timeseries=df_copy,
        _regional_attributes=regional_attributes,
        _provenance=timeseries._provenance,
    )

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
    df_copy = timeseries._timeseries.copy()
    grouped_df = timeseries.groupby_region()

    zscores = grouped_df[CommonFields.NEW_CASES].apply(_calculate_modified_zscore)

    to_exclude = (zscores > zscore_threshold) & (df_copy[CommonFields.NEW_CASES] > case_threshold)
    df_copy.loc[to_exclude, CommonFields.NEW_CASES] = None

    latest_values = _add_new_cases_to_latest(df_copy, timeseries._regional_attributes)
    new_timeseries = dataclasses.replace(
        timeseries, _timeseries=df_copy, _regional_attributes=latest_values
    )

    return new_timeseries


def drop_regions_without_population(
    mrts: MultiRegionDataset,
    known_location_id_to_drop: Sequence[str],
    log: Union[structlog.BoundLoggerBase, structlog._config.BoundLoggerLazyProxy],
) -> MultiRegionDataset:
    assert mrts._regional_attributes.index.names == [CommonFields.LOCATION_ID]
    latest_population = mrts._regional_attributes[CommonFields.POPULATION]
    locations_with_population = mrts._regional_attributes.loc[latest_population.notna()].index
    locations_without_population = mrts._regional_attributes.loc[latest_population.isna()].index
    unexpected_drops = set(locations_without_population) - set(known_location_id_to_drop)
    if unexpected_drops:
        log.warning(
            "Dropping unexpected regions without populaton", location_ids=sorted(unexpected_drops)
        )
    return mrts.get_locations_subset(locations_with_population)
