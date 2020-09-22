import datetime
import pathlib
from dataclasses import dataclass
from typing import Iterable
from typing import List, Optional, Union, TextIO
from typing import Sequence
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


@final
@dataclass(frozen=True)
class OneRegionTimeseriesDataset:
    """A set of timeseries with values from one region."""

    # Do not make an assumptions about a FIPS or location_id column in the DataFrame.
    data: pd.DataFrame
    # The region is not an attribute at this time because it simplifies making instances and
    # code that needs the region of an instance already has it.

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
            self.data.loc[rows_key, columns_key].reset_index(drop=True)
        )

    def remove_padded_nans(self, columns: List[str]):
        return OneRegionTimeseriesDataset(_remove_padded_nans(self.data, columns))


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
            .set_index([CommonFields.FIPS, "variable", CommonFields.DATE])
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
        assert summary.index.names == [CommonFields.FIPS, "variable"]
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


@final
@dataclass(frozen=True)
class MultiRegionTimeseriesDataset(SaveableDatasetInterface):
    """A set of timeseries with values from any number of regions."""

    # `data` may be used to process every row without considering the date or region. Keep logic about
    # FIPS/location_id/region containing in this class by using methods such as `get_one_region`. Do
    # *not* read date or region related columns directly from `data`.
    data: pd.DataFrame

    provenance: Optional[pd.Series] = None

    @property
    def dataset_type(self) -> DatasetType:
        return DatasetType.MULTI_REGION

    @classmethod
    def load_csv(cls, path_or_buf: Union[pathlib.Path, TextIO]):
        return MultiRegionTimeseriesDataset.from_csv(path_or_buf)

    @staticmethod
    def from_csv(path_or_buf: Union[pathlib.Path, TextIO]) -> "MultiRegionTimeseriesDataset":
        df = common_df.read_csv(path_or_buf, set_index=False)
        if CommonFields.LOCATION_ID not in df.columns:
            raise ValueError("MultiRegionTimeseriesDataset.from_csv requires location_id column")
        _add_fips_if_missing(df)
        return MultiRegionTimeseriesDataset(df)

    @staticmethod
    def from_timeseries(ts: TimeseriesDataset) -> "MultiRegionTimeseriesDataset":
        df = ts.data.copy()
        _add_location_id(df)

        if ts.provenance is not None:
            assert ts.provenance.index.names == [CommonFields.FIPS, "variable"]
            provenance = ts.provenance.copy()
            provenance.index = provenance.index.map(
                lambda i: (pipeline.fips_to_location_id(i[0]), i[1])
            )
        else:
            provenance = None

        return MultiRegionTimeseriesDataset(df, provenance=provenance)

    def get_one_region(self, region: Region) -> OneRegionTimeseriesDataset:
        rows_key = self.data[CommonFields.LOCATION_ID] == region.location_id
        return OneRegionTimeseriesDataset(data=self.data.loc[rows_key, :])

    def get_counties(
        self, after: Optional[datetime.datetime] = None
    ) -> "MultiRegionTimeseriesDataset":
        rows_key = dataset_utils.make_rows_key(
            self.data, aggregation_level=AggregationLevel.COUNTY, after=after
        )
        return MultiRegionTimeseriesDataset(self.data.loc[rows_key, :].reset_index(drop=True))

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
        self.to_timeseries().to_csv(path)


def _remove_padded_nans(df, columns):
    if df[columns].isna().all(axis=None):
        return df.loc[[False] * len(df), :].reset_index(drop=True)

    first_valid_index = min(df[column].first_valid_index() for column in columns)
    last_valid_index = max(df[column].last_valid_index() for column in columns)
    df = df.iloc[first_valid_index : last_valid_index + 1]
    return df.reset_index(drop=True)
