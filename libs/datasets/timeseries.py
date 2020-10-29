import dataclasses
import datetime
import pathlib
import warnings
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List, Optional, Union, TextIO
from typing import Sequence
from typing import Tuple

from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import PdFields
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

    # Do not make an assumptions about a FIPS or location_id column in the DataFrame.
    data: pd.DataFrame
    # The region is not an attribute at this time because it simplifies making instances and
    # code that needs the region of an instance already has it.

    latest: Dict[str, Any]

    # A default exists for convience in tests. Non-test could is expected to explicitly set
    # provenance.
    # Because these objects are frozen it /might/ be safe to use default={} but using a factory to
    # make a new instance of the mutable {} is safer.
    provenance: Dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if CommonFields.LOCATION_ID in self.data.columns:
            region_count = self.data[CommonFields.LOCATION_ID].nunique()
        else:
            region_count = self.data[CommonFields.FIPS].nunique()
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
        return OneRegionTimeseriesDataset(
            self.data.loc[rows_key, columns_key].reset_index(drop=True),
            latest=self.latest,
            provenance=self.provenance,
        )

    def remove_padded_nans(self, columns: List[str]):
        return OneRegionTimeseriesDataset(
            _remove_padded_nans(self.data, columns), latest=self.latest, provenance=self.provenance,
        )


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


def _index_latest_df(
    latest_df: pd.DataFrame, ts_locations: Union[pd.api.extensions.ExtensionArray, np.ndarray]
) -> pd.DataFrame:
    """Adds index to latest_df using location_id in ts_df."""
    assert latest_df.index.names == [None]

    if latest_df.empty:
        warnings.warn(BadMultiRegionWarning("Unexpected empty latest DataFrame"))
        return pd.DataFrame(index=ts_locations).sort_index()
    else:
        latest_df_with_index = latest_df.set_index(CommonFields.LOCATION_ID, verify_integrity=True)
        # Make an index with the union of the locations in the timeseries and latest_df to keep all rows of
        # latest_df
        all_locations = (
            latest_df_with_index.index.union(ts_locations)
            .unique()
            .sort_values()
            # Make sure the index has a name so that reset_index() restores the column name.
            .rename(CommonFields.LOCATION_ID)
        )
        # reindex takes the name from index `all_locations`, see
        # https://github.com/pandas-dev/pandas/issues/9885
        return latest_df_with_index.reindex(index=all_locations)


@final
@dataclass(frozen=True)
class MultiRegionTimeseriesDataset(SaveableDatasetInterface):
    """A set of timeseries and constant values from any number of regions."""

    # TODO(tom): rename to MultiRegionDataset

    # `data` may be used to process every row without considering the date or region. Keep logic about
    # FIPS/location_id/region containing in this class by using methods such as `get_one_region`. Do
    # *not* read date or region related columns directly from `data`. `data_with_fips` exists so we can
    # easily find code that reads the FIPS column.
    # `data` contains columns from CommonFields and simple integer index. DATE and LOCATION_ID must
    # be non-null in every row.
    data: pd.DataFrame

    # `latest_data` contains columns from CommonFields and a LOCATION_ID index.
    # If you need FIPS read from `latest_data_with_fips` so we can easily find code that depends on
    # the column.
    latest_data: pd.DataFrame

    # `provenance` is an array of str with a MultiIndex with names LOCATION_ID and VARIABLE.
    provenance: Optional[pd.Series] = None

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.MULTI_REGION

    @property
    def data_with_fips(self) -> pd.DataFrame:
        """data with FIPS column, use `data` when FIPS is not need."""
        return self.data

    @property
    def latest_data_with_fips(self) -> pd.DataFrame:
        """latest_data with FIPS column and LOCATION_ID index, use `latest_data` when FIPS is not need."""
        return self.latest_data

    @property
    def combined_df(self) -> pd.DataFrame:
        """"A DataFrame with timeseries data and latest data (with DATE=NaT) together."""
        return pd.concat([self.data, self.latest_data.reset_index()], ignore_index=True)

    @classmethod
    def load_csv(cls, path_or_buf: Union[pathlib.Path, TextIO]):
        return MultiRegionTimeseriesDataset.from_csv(path_or_buf)

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

    @staticmethod
    def from_timeseries_df(
        timeseries_df: pd.DataFrame, provenance: Optional[pd.Series] = None
    ) -> "MultiRegionTimeseriesDataset":
        assert timeseries_df.index.names == [None]
        assert CommonFields.LOCATION_ID in timeseries_df.columns
        empty_latest_df = pd.DataFrame([], index=pd.Index([], name=CommonFields.LOCATION_ID))
        return MultiRegionTimeseriesDataset(timeseries_df, empty_latest_df, provenance=provenance)

    def append_latest_df(self, latest_df: pd.DataFrame) -> "MultiRegionTimeseriesDataset":
        assert latest_df.index.names == [None]
        assert CommonFields.LOCATION_ID in latest_df.columns

        # test_top_level_metrics_basic depends on some empty columns being preserved in the
        # MultiRegionTimeseriesDataset so don't call dropna in this method.
        ts_locations = self.data[CommonFields.LOCATION_ID].unique()
        ts_locations.sort()
        latest_df = _index_latest_df(latest_df, ts_locations)
        common_columns = set(latest_df.columns) & set(self.latest_data.columns)
        if common_columns:
            warnings.warn(f"Common columns {common_columns}")
        latest_df = pd.concat([self.latest_data, latest_df], axis=1)

        return MultiRegionTimeseriesDataset(self.data, latest_df, provenance=self.provenance)

    def append_provenance_csv(
        self, path_or_buf: Union[pathlib.Path, TextIO]
    ) -> "MultiRegionTimeseriesDataset":
        df = pd.read_csv(path_or_buf)
        if PdFields.VALUE in df.columns:
            # Handle older CSV files that used 'value' header for provenance.
            df = df.rename(columns={PdFields.VALUE: PdFields.PROVENANCE})
        return self.append_provenance_df(df)

    def append_provenance_df(self, provenance_df: pd.DataFrame) -> "MultiRegionTimeseriesDataset":
        """Returns a new object containing data in self and given provenance information."""
        if self.provenance is not None:
            raise NotImplementedError("TODO(tom): add support for merging provenance data")
        assert provenance_df.index.names == [None]
        assert CommonFields.LOCATION_ID in provenance_df.columns
        assert PdFields.VARIABLE in provenance_df.columns
        assert PdFields.PROVENANCE in provenance_df.columns
        provenance_series = provenance_df.set_index([CommonFields.LOCATION_ID, PdFields.VARIABLE])[
            PdFields.PROVENANCE
        ]
        return MultiRegionTimeseriesDataset(
            self.data, latest_data=self.latest_data, provenance=provenance_series
        )

    @staticmethod
    def from_combined_dataframe(
        combined_df: pd.DataFrame, provenance: Optional[pd.Series] = None
    ) -> "MultiRegionTimeseriesDataset":
        """Builds a new object from a DataFrame containing timeseries and latest data.

        This method splits rows with DATE NaT into the latest values DataFrame, adds a FIPS column
        derived from LOCATION_ID, drops columns without data and calls `from_timeseries_df` to finish
        the construction.

        TODO(tom): This method isn't really useful beyond from_csv. Stop calling it directly and
        inline in from_csv.
        """
        assert combined_df.index.names == [None]
        if CommonFields.LOCATION_ID not in combined_df.columns:
            raise ValueError("MultiRegionTimeseriesDataset.from_csv requires location_id column")
        _add_fips_if_missing(combined_df)

        rows_with_date = combined_df[CommonFields.DATE].notna()
        timeseries_df = combined_df.loc[rows_with_date, :].dropna("columns", "all")

        # Extract rows of combined_df which don't have a date.
        latest_df = combined_df.loc[~rows_with_date, :].dropna("columns", "all")

        multiregion_timeseries = MultiRegionTimeseriesDataset.from_timeseries_df(
            timeseries_df, provenance=provenance
        )
        if not latest_df.empty:
            multiregion_timeseries = multiregion_timeseries.append_latest_df(latest_df)

        return multiregion_timeseries

    @staticmethod
    def from_csv(path_or_buf: Union[pathlib.Path, TextIO]) -> "MultiRegionTimeseriesDataset":
        dataset = MultiRegionTimeseriesDataset.from_combined_dataframe(
            common_df.read_csv(path_or_buf, set_index=False)
        )
        if isinstance(path_or_buf, pathlib.Path):
            provenance_path = pathlib.Path(str(path_or_buf).replace(".csv", "-provenance.csv"))
            if provenance_path.exists():
                dataset = dataset.append_provenance_csv(provenance_path)
        return dataset

    @staticmethod
    def from_timeseries_and_latest(
        ts: TimeseriesDataset, latest: LatestValuesDataset
    ) -> "MultiRegionTimeseriesDataset":
        """Converts legacy FIPS to new LOCATION_ID and calls `from_timeseries_df` to finish construction."""
        timeseries_df = ts.data.copy()
        _add_location_id(timeseries_df)

        latest_df = latest.data.copy()
        _add_location_id(latest_df)

        if ts.provenance is not None:
            # Check that current index is as expected. Names will be fixed after remapping, below.
            assert ts.provenance.index.names == [CommonFields.FIPS, PdFields.VARIABLE]
            provenance = ts.provenance.copy()
            provenance.index = provenance.index.map(
                lambda i: (pipeline.fips_to_location_id(i[0]), i[1])
            )
            provenance.index.rename([CommonFields.LOCATION_ID, PdFields.VARIABLE], inplace=True)
        else:
            provenance = None

        # TODO(tom): Either copy latest.provenance to its own series, blend with the timeseries
        # provenance (though some variable names are the same), or retire latest as a separate thing upstream.

        return MultiRegionTimeseriesDataset.from_timeseries_df(
            timeseries_df, provenance=provenance
        ).append_latest_df(latest_df)

    def __post_init__(self):
        # Some integrity checks
        assert CommonFields.LOCATION_ID in self.data.columns
        assert self.data[CommonFields.LOCATION_ID].notna().all()
        assert self.data.index.is_unique
        assert self.data.index.is_monotonic_increasing
        assert self.latest_data.index.names == [CommonFields.LOCATION_ID]
        if self.provenance is not None:
            assert isinstance(self.provenance, pd.Series)
            assert self.provenance.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]

    def append_regions(
        self, other: "MultiRegionTimeseriesDataset"
    ) -> "MultiRegionTimeseriesDataset":
        return MultiRegionTimeseriesDataset.from_timeseries_df(
            pd.concat([self.data, other.data], ignore_index=True)
        ).append_latest_df(
            pd.concat(
                [self.latest_data.reset_index(), other.latest_data.reset_index()], ignore_index=True
            )
        )

    def get_one_region(self, region: Region) -> OneRegionTimeseriesDataset:
        ts_df = self.data.loc[self.data[CommonFields.LOCATION_ID] == region.location_id, :]
        latest_dict = self._location_id_latest_dict(region.location_id)
        if ts_df.empty and not latest_dict:
            raise RegionLatestNotFound(region)
        provenance_dict = self._location_id_provenance_dict(region.location_id)

        return OneRegionTimeseriesDataset(
            data=ts_df, latest=latest_dict, provenance=provenance_dict,
        )

    def _location_id_provenance_dict(self, location_id: str) -> dict:
        """Returns the provenance dict of a location_id."""
        if self.provenance is None:
            return {}
        try:
            provenance_series = self.provenance.loc[location_id]
        except KeyError:
            return {}
        else:
            return provenance_series[provenance_series.notna()].to_dict()

    def _location_id_latest_dict(self, location_id: str) -> dict:
        """Returns the latest values dict of a location_id."""
        try:
            latest_row = self.latest_data.loc[location_id, :]
        except KeyError:
            latest_row = pd.Series([], dtype=object)
        return latest_row.where(pd.notnull(latest_row), None).to_dict()

    def get_regions_subset(self, regions: Sequence[Region]) -> "MultiRegionTimeseriesDataset":
        location_ids = pd.Index(sorted(r.location_id for r in regions))
        return self.get_locations_subset(location_ids)

    def get_locations_subset(self, location_ids: Sequence[str]) -> "MultiRegionTimeseriesDataset":
        timeseries_df = self.data.loc[self.data[CommonFields.LOCATION_ID].isin(location_ids), :]
        latest_df, provenance = self._get_latest_and_provenance_for_locations(location_ids)
        return MultiRegionTimeseriesDataset.from_timeseries_df(
            timeseries_df, provenance=provenance
        ).append_latest_df(latest_df)

    def get_counties(
        self, after: Optional[datetime.datetime] = None
    ) -> "MultiRegionTimeseriesDataset":
        ts_rows_key = dataset_utils.make_rows_key(
            self.data, aggregation_level=AggregationLevel.COUNTY, after=after
        )
        ts_df = self.data.loc[ts_rows_key, :].reset_index(drop=True)

        location_ids = ts_df[CommonFields.LOCATION_ID].unique()

        # `after` doesn't make sense for latest because it doesn't contain date information.
        # Keep latest data for regions that are in the new timeseries DataFrame.
        # TODO(tom): Replace this with re-calculating latest data from the new timeseries so that
        # metrics which no longer have real values are excluded.
        latest_df, provenance = self._get_latest_and_provenance_for_locations(location_ids)

        return MultiRegionTimeseriesDataset.from_combined_dataframe(
            pd.concat([ts_df, latest_df], ignore_index=True), provenance=provenance
        )

    def _get_latest_and_provenance_for_locations(
        self, location_ids
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        latest_df = self.latest_data.loc[
            self.latest_data.index.get_level_values(CommonFields.LOCATION_ID).isin(location_ids), :
        ].reset_index()
        if self.provenance is not None:
            provenance = self.provenance[
                self.provenance.index.get_level_values(CommonFields.LOCATION_ID).isin(location_ids)
            ]
        else:
            provenance = None
        return latest_df, provenance

    def groupby_region(self) -> pandas.core.groupby.generic.DataFrameGroupBy:
        return self.data.groupby(CommonFields.LOCATION_ID)

    @property
    def empty(self) -> bool:
        return self.data.empty

    def to_timeseries(self) -> TimeseriesDataset:
        """Returns a `TimeseriesDataset` of this data.

        This method exists for interim use when calling code that doesn't use MultiRegionTimeseriesDataset.
        """
        return TimeseriesDataset(self.data, provenance=self.provenance)

    def to_csv(self, path: pathlib.Path):
        """Persists timeseries to CSV.

        Args:
            path: Path to write to.
        """
        combined = self.combined_df
        assert combined[CommonFields.LOCATION_ID].notna().all()
        common_df.write_csv(
            combined, path, structlog.get_logger(), [CommonFields.LOCATION_ID, CommonFields.DATE]
        )
        if self.provenance is not None:
            provenance_path = str(path).replace(".csv", "-provenance.csv")
            self.provenance.sort_index().rename(PdFields.PROVENANCE).to_csv(provenance_path)

    def join_columns(self, other: "MultiRegionTimeseriesDataset") -> "MultiRegionTimeseriesDataset":
        """Joins the timeseries columns in `other` with those in `self`."""
        if not other.latest_data.empty:
            raise NotImplementedError("No support for joining other with latest_data")
        other_df = other.data_with_fips.set_index([CommonFields.LOCATION_ID, CommonFields.DATE])
        self_df = self.data_with_fips.set_index([CommonFields.LOCATION_ID, CommonFields.DATE])
        other_geo_columns = set(other_df.columns) & set(GEO_DATA_COLUMNS)
        other_ts_columns = (
            set(other_df.columns) - set(GEO_DATA_COLUMNS) - set(TimeseriesDataset.INDEX_FIELDS)
        )
        common_ts_columns = other_ts_columns & set(self.data_with_fips.columns)
        if common_ts_columns:
            # columns to be joined need to be disjoint
            raise ValueError(f"Columns are in both dataset: {common_ts_columns}")
        common_geo_columns = list(set(self.data_with_fips.columns) & other_geo_columns)
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
        combined_df = pd.concat([self_df, other_df[list(other_ts_columns)]], axis=1)
        if self.provenance is not None:
            if other.provenance is not None:
                combined_provenance = pd.concat([self.provenance, other.provenance])
            else:
                combined_provenance = self.provenance
        else:
            combined_provenance = other.provenance
        return MultiRegionTimeseriesDataset.from_timeseries_df(
            combined_df.reset_index(), provenance=combined_provenance,
        ).append_latest_df(self.latest_data_with_fips.reset_index())

    def iter_one_regions(self) -> Iterable[Tuple[Region, OneRegionTimeseriesDataset]]:
        """Iterates through all the regions in this object"""
        for location_id, data_group in self.data_with_fips.groupby(CommonFields.LOCATION_ID):
            latest_dict = self._location_id_latest_dict(location_id)
            provenance_dict = self._location_id_provenance_dict(location_id)
            yield Region(location_id=location_id, fips=None), OneRegionTimeseriesDataset(
                data_group, latest_dict, provenance=provenance_dict,
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
    return latest_copy.reset_index()


def add_new_cases(timeseries: MultiRegionTimeseriesDataset) -> MultiRegionTimeseriesDataset:
    """Adds a new_cases column to this dataset by calculating the daily diff in cases."""
    df_copy = timeseries.data.copy()
    grouped_df = timeseries.groupby_region()
    # Calculating new cases using diff will remove the first detected value from the case series.
    # We want to capture the first day a region reports a case. Since our data sources have
    # been capturing cases in all states from the beginning of the pandemic, we are treating
    # The first days as appropriate new case data.
    new_cases = grouped_df[CommonFields.CASES].apply(_diff_preserving_first_value)

    # Remove the occasional negative case adjustments.
    new_cases[new_cases < 0] = pd.NA

    df_copy[CommonFields.NEW_CASES] = new_cases
    latest_values = _add_new_cases_to_latest(df_copy, timeseries.latest_data)

    new_timeseries = MultiRegionTimeseriesDataset.from_timeseries_df(
        timeseries_df=df_copy, provenance=timeseries.provenance
    ).append_latest_df(latest_values)

    return new_timeseries


def drop_regions_without_population(
    mrts: MultiRegionTimeseriesDataset,
    known_location_id_to_drop: Sequence[str],
    log: Union[structlog.BoundLoggerBase, structlog._config.BoundLoggerLazyProxy],
) -> MultiRegionTimeseriesDataset:
    # latest_population is a Series with location_id index
    latest_population = mrts.latest_data[CommonFields.POPULATION]
    locations_with_population = mrts.latest_data.loc[latest_population.notna()].index
    locations_without_population = mrts.latest_data.loc[latest_population.isna()].index
    unexpected_drops = set(locations_without_population) - set(known_location_id_to_drop)
    if unexpected_drops:
        log.warning(
            "Dropping unexpected regions without populaton", location_ids=sorted(unexpected_drops)
        )
    return mrts.get_locations_subset(locations_with_population)
