from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    TextIO,
    Mapping,
    Set,
    Sequence,
    Tuple,
)
import dataclasses
import datetime
import pathlib
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from collections import defaultdict

import more_itertools
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.common import is_bool_dtype
from typing_extensions import final

import pandas as pd
import numpy as np
import structlog
from covidactnow.datapublic import common_df
from libs import pipeline
from libs.dataclass_utils import dataclass_with_default_init
from libs.datasets import dataset_pointer
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.dataset_utils import GEO_DATA_COLUMNS
from libs.datasets.dataset_utils import NON_NUMERIC_COLUMNS
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets import taglib
from libs.datasets import demographics
from libs.datasets.taglib import TagField
from libs.datasets.taglib import TagType
from libs.datasets.taglib import UrlStr
from libs.datasets.demographics import DistributionBucket
from libs.pipeline import Region
import pandas.core.groupby.generic
from backports.cached_property import cached_property

from libs.pipeline import RegionMaskOrRegion


_log = structlog.get_logger()


# Fields used as panda MultiIndex levels when tags are represented in a pd.Series
_TAG_INDEX_FIELDS = [
    TagField.LOCATION_ID,
    TagField.VARIABLE,
    TagField.DEMOGRAPHIC_BUCKET,
    TagField.TYPE,
]

_TAG_DF_COLUMNS = _TAG_INDEX_FIELDS + [TagField.CONTENT]

ANNOTATION_TAG_TYPES = [
    TagType.CUMULATIVE_LONG_TAIL_TRUNCATED,
    TagType.CUMULATIVE_TAIL_TRUNCATED,
    TagType.ZSCORE_OUTLIER,
]


# An empty pd.Series with the structure expected for the tag attribute. Use this when a
# dataset does not have any tags.
_EMPTY_TAG_SERIES = pd.Series(
    [],
    name=TagField.CONTENT,
    dtype="str",
    index=pd.MultiIndex.from_tuples([], names=_TAG_INDEX_FIELDS),
)
_EMPTY_ONE_REGION_TAG_SERIES = _EMPTY_TAG_SERIES.droplevel(TagField.LOCATION_ID)


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

    bucketed_latest: pd.DataFrame

    # A default exists for convenience in tests. Non-test code is expected to explicitly set tag.
    tag: pd.Series = _EMPTY_ONE_REGION_TAG_SERIES

    @cached_property
    def tag_all_bucket(self) -> pd.Series:
        try:
            return self.tag.xs(DemographicBucket.ALL, level=TagField.DEMOGRAPHIC_BUCKET)
        except KeyError:
            return _EMPTY_ONE_REGION_TAG_SERIES.droplevel([TagField.DEMOGRAPHIC_BUCKET])

    @property
    def provenance(self) -> Mapping[CommonFields, List[str]]:
        provenance_series = self.tag_all_bucket.loc[:, [TagType.PROVENANCE]].droplevel(
            [TagField.TYPE]
        )
        # https://stackoverflow.com/a/56065318
        return provenance_series.groupby(level=0).agg(list).to_dict()

    @property
    def source_url(self) -> Mapping[CommonFields, List[UrlStr]]:
        source_url_series = self.tag_all_bucket.loc[:, [TagType.SOURCE_URL]].droplevel(
            [TagField.TYPE]
        )
        # https://stackoverflow.com/a/56065318
        return source_url_series.groupby(level=0).agg(list).to_dict()

    def annotations_all_bucket(self, metric: FieldName) -> List[taglib.AnnotationWithDate]:
        try:
            return self.tag_objects_series.loc[
                [metric], DemographicBucket.ALL, ANNOTATION_TAG_TYPES
            ].to_list()
        except KeyError:
            # Not very elegant but I can't find
            # anything better in https://github.com/pandas-dev/pandas/issues/10695
            return []

    def sources_all_bucket(self, field_name: FieldName) -> List[taglib.Source]:
        try:
            return self.tag_objects_series.loc[
                [field_name], DemographicBucket.ALL, [TagType.SOURCE]
            ].to_list()
        except KeyError:
            return []

    @cached_property
    def demographic_distributions_by_field(
        self,
    ) -> Dict[CommonFields, Dict[str, demographics.ScalarDistribution]]:
        """Returns demographic distributions by field.

        Only returns distributions where at least one value exists.
        """
        result = defaultdict(lambda: defaultdict(dict))
        bucketed_latest = self.bucketed_latest.where(pd.notnull(self.bucketed_latest), None)

        for field in bucketed_latest.columns:
            field_data = bucketed_latest.loc[:, field].to_dict()
            for short_name, value in field_data.items():
                # Skipping 'all' it as it does not have multiple values per distribution.
                if short_name == "all":
                    continue

                # Only adding real-valued data to distributions.  If there are multiple
                # variables with different buckets for a given distribution, including none's
                # will mix buckets.  To properly fix, consider passing latest in a long format
                # rather than a wide variables format.
                if value is None:
                    continue
                bucket = DistributionBucket.from_str(short_name)
                result[field][bucket.distribution][bucket.name] = value

        return result

    @cached_property
    def tag_objects_series(self) -> pd.Series:
        """A Series of TagInTimeseries objects, indexed like self.tag for easy lookups."""
        assert self.tag.index.names[2] == TagField.TYPE
        # Apply a function to each element in the Series self.tag with the function having access to
        # the index of each element. From https://stackoverflow.com/a/47645833/341400.
        # result_type reduce forces the return value to be a Series, even when tag is empty.
        return self.tag.to_frame().apply(
            lambda row: taglib.TagInTimeseries.make(row.name[2], content=row.content),
            axis=1,
            result_type="reduce",
        )

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

        assert self.tag.index.names == [
            TagField.VARIABLE,
            TagField.DEMOGRAPHIC_BUCKET,
            TagField.TYPE,
        ]

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
    df[CommonFields.LOCATION_ID] = df[CommonFields.FIPS].map(dataset_utils.get_fips_to_location())
    # Missing values are mapped to NaN but we want it to raise. There doesn't seem to be a better
    # work-around at https://github.com/pandas-dev/pandas/issues/14210
    if df[CommonFields.LOCATION_ID].isna().any():
        raise KeyError(
            f"No location_id found for "
            f"{df.loc[df[CommonFields.LOCATION_ID].isna(), CommonFields.FIPS].value_counts()}"
        )
    return df


class ExtraColumnWarning(UserWarning):
    pass


def _map_and_warn_about_mismatches(timeseries_df: pd.DataFrame, field_name: FieldName):
    """Warns if field_name differs from the static geo data for `field_name`."""
    COMPUTED = FieldName("computed")
    orig_series = timeseries_df.loc(axis=1)[field_name]
    map_from_location_id = dataset_utils.get_geo_data()[field_name]
    computed_series = orig_series.index.get_level_values(CommonFields.LOCATION_ID).map(
        map_from_location_id
    )
    df = pd.DataFrame({COMPUTED: computed_series, field_name: orig_series})
    # If timeseries_df didn't have a value for aggregate level don't compare it to the
    # computed value.
    df = df.dropna(subset=[field_name])
    bad_df = df.loc[df[COMPUTED] != df[field_name]]
    if not bad_df.empty:
        warnings.warn(ExtraColumnWarning(f"Bad {field_name}\n{bad_df}"), stacklevel=2)


def _warn_and_drop_extra_columns(timeseries_df: pd.DataFrame) -> pd.DataFrame:
    # TODO(tom): After tests are cleaned up to not include these extra columns, most likely by
    #  building MultiRegionDataset using test_helpers instead of parsing a CSV, remove these checks.
    if CommonFields.FIPS in timeseries_df.columns:
        _map_and_warn_about_mismatches(timeseries_df, CommonFields.FIPS)
        timeseries_df = timeseries_df.drop(columns=CommonFields.FIPS)
    if CommonFields.AGGREGATE_LEVEL in timeseries_df.columns:
        _map_and_warn_about_mismatches(timeseries_df, CommonFields.AGGREGATE_LEVEL)
        timeseries_df = timeseries_df.drop(columns=CommonFields.AGGREGATE_LEVEL)
    if CommonFields.STATE in timeseries_df.columns:
        _map_and_warn_about_mismatches(timeseries_df, CommonFields.STATE)
        timeseries_df = timeseries_df.drop(columns=CommonFields.STATE)
    timeseries_df = timeseries_df.drop(columns=CommonFields.COUNTY, errors="ignore")
    geodata_column_mask = timeseries_df.columns.isin(
        set(TIMESERIES_INDEX_FIELDS) | set(GEO_DATA_COLUMNS)
    )
    if geodata_column_mask.any():
        warnings.warn(
            ExtraColumnWarning(
                f"Ignoring extra columns: " f"{timeseries_df.columns[geodata_column_mask]}"
            ),
            stacklevel=2,
        )
    timeseries_df = timeseries_df.loc[:, ~geodata_column_mask]
    return timeseries_df


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


def _add_aggregate_level_if_missing(df: pd.DataFrame):
    """Adds the aggregate level column if missing, in place."""
    assert CommonFields.LOCATION_ID in df.columns

    if CommonFields.AGGREGATE_LEVEL not in df.columns:
        df[CommonFields.AGGREGATE_LEVEL] = df[CommonFields.LOCATION_ID].apply(
            lambda x: Region.from_location_id(x).level.value
        )


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


# An empty pd.DataFrame with the structure expected for the static attribute. Use this when
# a dataset does not have any static values.
EMPTY_STATIC_DF = pd.DataFrame(
    [],
    index=pd.Index([], name=CommonFields.LOCATION_ID),
    columns=pd.Index([], name=PdFields.VARIABLE),
)


# An empty DataFrame with the expected index names for a timeseries with row labels <location_id,
# variable, bucket> and column labels <date>.
EMPTY_TIMESERIES_BUCKETED_WIDE_DATES_DF = pd.DataFrame(
    [],
    dtype="float",
    index=pd.MultiIndex.from_tuples(
        [], names=[CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET]
    ),
    columns=pd.DatetimeIndex([], name=CommonFields.DATE),
)


EMPTY_TIMESERIES_NOT_BUCKETED_WIDE_DATES_DF = EMPTY_TIMESERIES_BUCKETED_WIDE_DATES_DF.droplevel(
    PdFields.DEMOGRAPHIC_BUCKET
)


# An empty DataFrame with the expected index names for a timeseries with row labels <location_id,
# date> and column labels <variable>. This is the structure of most CSV files in this repo as of
# Nov 2020.
EMPTY_TIMESERIES_WIDE_VARIABLES_DF = pd.DataFrame(
    [],
    dtype="float",
    index=pd.MultiIndex.from_tuples([], names=[CommonFields.LOCATION_ID, CommonFields.DATE]),
    columns=pd.Index([], name=PdFields.VARIABLE),
)

EMPTY_TIMESERIES_BUCKETED_WIDE_VARIABLES_DF = pd.DataFrame(
    [],
    dtype="float",
    index=pd.MultiIndex.from_tuples(
        [], names=[CommonFields.LOCATION_ID, PdFields.DEMOGRAPHIC_BUCKET, CommonFields.DATE]
    ),
    columns=pd.Index([], name=PdFields.VARIABLE),
)


def _check_timeseries_wide_vars_index(timeseries_index: pd.MultiIndex, *, bucketed: bool):
    if bucketed:
        assert timeseries_index.names == [
            CommonFields.LOCATION_ID,
            PdFields.DEMOGRAPHIC_BUCKET,
            CommonFields.DATE,
        ]
    else:
        # timeseries.index order is important for _timeseries_latest_values correctness.
        assert timeseries_index.names == [CommonFields.LOCATION_ID, CommonFields.DATE]
    assert timeseries_index.is_unique
    assert timeseries_index.is_monotonic_increasing


def _check_timeseries_wide_vars_structure(wide_vars_df: pd.DataFrame, *, bucketed: bool):
    """Asserts that a DataFrame has the structure expected with wide-variable columns."""
    _check_timeseries_wide_vars_index(wide_vars_df.index, bucketed=bucketed)
    assert wide_vars_df.columns.names == [PdFields.VARIABLE]
    numeric_columns = wide_vars_df.dtypes.apply(is_numeric_dtype)
    assert numeric_columns.all()


def _tag_add_all_bucket(tag: pd.Series) -> pd.Series:
    tag_bucketed = pd.concat(
        {DemographicBucket.ALL: tag}, names=[PdFields.DEMOGRAPHIC_BUCKET]
    ).reorder_levels(_TAG_INDEX_FIELDS)
    return tag_bucketed


def tag_df_add_all_bucket_in_place(tag_df: pd.DataFrame):
    tag_df[TagField.DEMOGRAPHIC_BUCKET] = "all"


# eq=False because instances are large and we want to compare by id instead of value
@final
@dataclass_with_default_init(frozen=True, eq=False)
class MultiRegionDataset:
    """A set of timeseries and static values from any number of regions.

    While the data may be accessed directly in the attributes `timeseries_bucketed`, `static` and
    `provenance` for easier future refactoring try to use (adding if not available) higher level
    methods that derive the data you need from these attributes.

    Methods named `append_...` return a new object with more regions of data. Methods named `add_...` and
    `join_...` return a new object with more data about the same regions, such as new metrics and provenance
    information.
    """

    # Timeseries metrics with float values. Each timeseries is identified by a variable name,
    # region and demographic bucket.
    timeseries_bucketed: pd.DataFrame

    # Static data, each identified by variable name and region. This includes county name,
    # state etc (GEO_DATA_COLUMNS) and metrics that change so slowly they can be
    # considered constant, such as population and hospital beds.
    static: pd.DataFrame = EMPTY_STATIC_DF

    # A Series of tag CONTENT values having index with levels TAG_INDEX_FIELDS (LOCATION_ID,
    # VARIABLE, TYPE). Rows with identical index values may exist.
    tag: pd.Series = _EMPTY_TAG_SERIES

    # noinspection PyMissingConstructor
    def __init__(
        self,
        *,
        timeseries: Optional[pd.DataFrame] = None,
        timeseries_bucketed: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        # TODO(tom): Replace all use of `timeseries` (not bucketed) with timeseries_bucketed,
        #  then remove this branch and the timeseries cached_property.
        if timeseries is not None:
            assert timeseries_bucketed is None
            _check_timeseries_wide_vars_structure(timeseries, bucketed=False)
            timeseries_bucketed = pd.concat(
                {DemographicBucket("all"): timeseries}, names=[PdFields.DEMOGRAPHIC_BUCKET]
            ).reorder_levels(EMPTY_TIMESERIES_BUCKETED_WIDE_VARIABLES_DF.index.names)

        self.__default_init__(  # pylint: disable=E1101
            timeseries_bucketed=timeseries_bucketed, **kwargs,
        )

    @cached_property
    def timeseries(self) -> pd.DataFrame:
        """Timeseries metrics with float values. Each timeseries is identified by a variable name
       and region"""
        try:
            return self.timeseries_bucketed.xs("all", level=PdFields.DEMOGRAPHIC_BUCKET, axis=0)
        except KeyError:
            # Return a DataFrame that has an index with no rows (but expected level names) and
            # columns copied from the input.
            return pd.DataFrame(
                [],
                index=EMPTY_TIMESERIES_WIDE_VARIABLES_DF.index,
                columns=self.timeseries_bucketed.columns,
                dtype="float",
            )

    @cached_property
    def tag_all_bucket(self) -> pd.Series:
        try:
            return self.tag.xs("all", level=TagField.DEMOGRAPHIC_BUCKET)
        except KeyError:
            return _EMPTY_TAG_SERIES.droplevel(TagField.DEMOGRAPHIC_BUCKET)

    @cached_property
    def location_ids(self) -> pd.Index:
        return self.static.index.unique(CommonFields.LOCATION_ID).union(
            self.timeseries_bucketed.index.unique(CommonFields.LOCATION_ID)
        )

    @cached_property
    def geo_data(self) -> pd.DataFrame:
        location_ids = self.location_ids
        geo_data = dataset_utils.get_geo_data()
        missing_location_id = location_ids.difference(geo_data.index)
        if not missing_location_id.empty:
            raise KeyError(f"location_id not in data/geo-data.csv:\n{missing_location_id}")
        return geo_data.reindex(index=location_ids)

    @cached_property
    def static_and_geo_data(self) -> pd.DataFrame:
        return self.static.join(self.geo_data)

    @property
    def timeseries_regions(self) -> Set[Region]:
        """Returns a set of all regions in the timeseries dataset."""

        location_ids = self.timeseries.index.get_level_values(CommonFields.LOCATION_ID)
        return set(Region.from_location_id(location_id) for location_id in location_ids)

    @cached_property
    def provenance(self) -> pd.DataFrame:
        """A Series of str with a MultiIndex with names LOCATION_ID and VARIABLE"""
        # TODO(tom): Remove this function. It is only used in tests.
        provenance_tags = self.tag_all_bucket.loc[:, :, [TagType.PROVENANCE]]
        return provenance_tags.droplevel([TagField.TYPE]).rename(PdFields.PROVENANCE)

    @cached_property
    def dataset_type(self) -> DatasetType:
        return DatasetType.MULTI_REGION

    @lru_cache(maxsize=None)
    def static_and_timeseries_latest_with_fips(self) -> pd.DataFrame:
        """Static values merged with the latest timeseries values."""
        return _merge_attributes(
            self._timeseries_latest_values().reset_index(), self.static_and_geo_data.reset_index()
        )

    @cached_property
    def timeseries_bucketed_long(self) -> pd.Series:
        """A Series with MultiIndex LOCATION_ID, DEMOGRAPHIC_BUCKET, DATE, VARIABLE"""
        return self.timeseries_bucketed.stack(dropna=True).rename(PdFields.VALUE).sort_index()

    @cached_property
    def wide_var_not_null(self) -> pd.DataFrame:
        """True iff there is at least one real value in any bucket with given location and
        variable"""
        wide_var_not_null = (
            self.timeseries_bucketed_long.notnull()
            .groupby([CommonFields.LOCATION_ID, PdFields.VARIABLE], sort=False)
            .any()
            # TODO(tom) unstack sometimes return dtype Object, sometimes bool. Work out why and
            #  remove astype.
            .unstack(PdFields.VARIABLE, fill_value=False)
            .sort_index()
            .astype(bool)
        )
        assert wide_var_not_null.index.names == [CommonFields.LOCATION_ID]
        assert wide_var_not_null.columns.names == [PdFields.VARIABLE]
        assert wide_var_not_null.dtypes.apply(is_bool_dtype).all()
        return wide_var_not_null

    @cached_property
    def timeseries_bucketed_wide_dates(self) -> pd.DataFrame:
        """Returns the timeseries in a DataFrame with LOCATION_ID, VARIABLE, BUCKET index and DATE
        columns."""
        if self.timeseries_bucketed.empty:
            return EMPTY_TIMESERIES_BUCKETED_WIDE_DATES_DF
        timeseries_long = self.timeseries_bucketed_long
        dates = timeseries_long.index.unique(CommonFields.DATE)
        if dates.empty:
            return EMPTY_TIMESERIES_BUCKETED_WIDE_DATES_DF
        start_date = dates.min()
        end_date = dates.max()
        date_range = pd.date_range(start=start_date, end=end_date)
        timeseries_wide = (
            timeseries_long.unstack(CommonFields.DATE)
            .reindex(columns=date_range)
            .rename_axis(columns=CommonFields.DATE)
            .reorder_levels(
                [CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET]
            )
        )
        if not timeseries_wide.columns.is_all_dates:
            raise ValueError(f"Problem with {start_date} to {end_date}... {str(self.timeseries)}")
        return timeseries_wide

    def print_stats(self, name: str):
        """Print stats about buckets by CommonFields group; quick and ugly"""
        buckets = self.timeseries_bucketed_long.index.get_level_values(PdFields.DEMOGRAPHIC_BUCKET)
        bucket_all = buckets == DemographicBucket.ALL
        count = (
            pd.DataFrame(
                {"all": bucket_all, "not_all": ~bucket_all},
                index=self.timeseries_bucketed_long.index.get_level_values(PdFields.VARIABLE),
            )
            .groupby(common_fields.COMMON_FIELD_TO_GROUP, sort=False)
            .sum()
        )
        print(f"Dataset {name}:\n{count}")

    @cached_property
    def timeseries_not_bucketed_wide_dates(self) -> pd.DataFrame:
        """Returns the timeseries in a DataFrame with LOCATION_ID, VARIABLE index and DATE columns."""
        # TODO(tom): Replace all calls to this function with calls to timeseries_bucketed_wide_dates
        try:
            return self.timeseries_bucketed_wide_dates.xs(
                "all", level=PdFields.DEMOGRAPHIC_BUCKET, axis=0
            )
        except KeyError:
            return EMPTY_TIMESERIES_NOT_BUCKETED_WIDE_DATES_DF

    def _timeseries_latest_values(self) -> pd.DataFrame:
        """Returns the latest value for every region and metric, derived from timeseries."""

        try:
            return self._timeseries_bucketed_latest_values.xs(
                "all", level=PdFields.DEMOGRAPHIC_BUCKET, axis=0
            )
        except KeyError:
            return pd.DataFrame([], index=pd.Index([], name=CommonFields.LOCATION_ID))

    @cached_property
    def _timeseries_bucketed_latest_values(self) -> pd.DataFrame:
        if self.timeseries.columns.empty:
            return pd.DataFrame(
                [],
                index=pd.MultiIndex.from_tuples(
                    [], names=[CommonFields.LOCATION_ID, PdFields.DEMOGRAPHIC_BUCKET]
                ),
            )
        long_bucketed = self.timeseries_bucketed.stack().sort_index().droplevel(CommonFields.DATE)
        unduplicated_bucketed_and_last_mask = ~long_bucketed.index.duplicated(keep="last")
        return long_bucketed.loc[unduplicated_bucketed_and_last_mask, :].unstack(PdFields.VARIABLE)

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
    def from_timeseries_wide_dates_df(
        timeseries_wide_dates: pd.DataFrame, *, bucketed=False
    ) -> "MultiRegionDataset":
        """Make a new dataset from a DataFrame as returned by timeseries_wide_dates."""
        if bucketed:
            assert timeseries_wide_dates.index.names == [
                CommonFields.LOCATION_ID,
                PdFields.VARIABLE,
                PdFields.DEMOGRAPHIC_BUCKET,
            ]
        else:
            assert timeseries_wide_dates.index.names == [
                CommonFields.LOCATION_ID,
                PdFields.VARIABLE,
            ]
        assert timeseries_wide_dates.columns.names == [CommonFields.DATE]
        timeseries_wide_dates.columns: pd.DatetimeIndex = pd.to_datetime(
            timeseries_wide_dates.columns
        )
        timeseries_wide_variables = (
            timeseries_wide_dates.stack().unstack(PdFields.VARIABLE).sort_index()
        )
        if bucketed:
            return MultiRegionDataset(timeseries_bucketed=timeseries_wide_variables)
        else:
            return MultiRegionDataset(timeseries=timeseries_wide_variables)

    @staticmethod
    def from_timeseries_df(timeseries_df: pd.DataFrame) -> "MultiRegionDataset":
        """Make a new dataset from a DataFrame containing timeseries (real-valued metrics)."""
        assert timeseries_df.index.names == [None]
        timeseries_df = timeseries_df.set_index(
            [CommonFields.LOCATION_ID, CommonFields.DATE]
        ).rename_axis(columns=PdFields.VARIABLE)

        timeseries_df = _warn_and_drop_extra_columns(timeseries_df)
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

        return MultiRegionDataset(timeseries=timeseries_df)

    def add_fips_static_df(self, latest_df: pd.DataFrame) -> "MultiRegionDataset":
        latest_df = _add_location_id(latest_df)
        return self.add_static_values(latest_df)

    def add_static_values(self, attributes_df: pd.DataFrame) -> "MultiRegionDataset":
        """Returns a new object with non-NA values in `latest_df` added to the static attribute."""
        scalars_without_geodata = attributes_df.loc[
            :, attributes_df.columns.difference(GEO_DATA_COLUMNS)
        ]
        combined_attributes = _merge_attributes(self.static.reset_index(), scalars_without_geodata)
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
                self.timeseries_not_bucketed_wide_dates.index, fill_value=provenance
            )
        )

    def add_tag_all_bucket(self, tag: taglib.TagInTimeseries) -> "MultiRegionDataset":
        """Returns a new object with given tag copied for every timeseries with bucket "all"."""
        tag_df = pd.DataFrame(
            {taglib.TagField.CONTENT: tag.content, taglib.TagField.TYPE: tag.tag_type},
            index=self.timeseries_not_bucketed_wide_dates.index,
        ).reset_index()
        tag_df_add_all_bucket_in_place(tag_df)
        return self.append_tag_df(tag_df)

    def add_provenance_series(self, provenance: pd.Series) -> "MultiRegionDataset":
        """Returns a new object containing data in self and given provenance information."""
        if not self.provenance.empty:
            raise NotImplementedError("TODO(tom): add support for merging provenance data")
        assert provenance.index.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
        assert isinstance(provenance, pd.Series)

        new_index_df = provenance.index.to_frame()
        new_index_df[TagField.TYPE] = TagType.PROVENANCE
        new_index_df[TagField.DEMOGRAPHIC_BUCKET] = DemographicBucket.ALL
        tag_additions = provenance.copy()
        tag_additions.index = pd.MultiIndex.from_frame(new_index_df).reorder_levels(
            _TAG_INDEX_FIELDS
        )
        # Make a sorted series. The order doesn't matter and sorting makes the order depend only on
        # what is represented, not the order it appears in the input.
        tag = pd.concat([self.tag, tag_additions]).sort_index().rename(TagField.CONTENT)
        return dataclasses.replace(self, tag=tag)

    @staticmethod
    def from_csv(path_or_buf: Union[pathlib.Path, TextIO]) -> "MultiRegionDataset":
        combined_df = common_df.read_csv(path_or_buf, set_index=False).rename_axis(
            columns=PdFields.VARIABLE
        )
        if CommonFields.LOCATION_ID not in combined_df.columns:
            raise ValueError("MultiRegionDataset.from_csv requires location_id column")

        # Split rows with DATE NaT into latest_df and call `from_timeseries_df` to finish the
        # construction.
        rows_with_date = combined_df[CommonFields.DATE].notna()
        timeseries_df = combined_df.loc[rows_with_date, :]

        # Extract rows of combined_df which don't have a date.
        latest_df = combined_df.loc[~rows_with_date, :].drop(columns=[CommonFields.DATE])

        dataset = MultiRegionDataset.from_timeseries_df(timeseries_df)
        if not latest_df.empty:
            dataset = dataset.add_static_values(latest_df)

        if isinstance(path_or_buf, pathlib.Path):
            provenance_path = pathlib.Path(str(path_or_buf).replace(".csv", "-provenance.csv"))
            if provenance_path.exists():
                # TODO(tom): Try to delete add_provenance_csv which seems to be only used in tests.
                dataset = dataset.add_provenance_csv(provenance_path)
        return dataset

    @staticmethod
    def read_from_pointer(pointer: dataset_pointer.DatasetPointer) -> "MultiRegionDataset":
        wide_dates_df = pd.read_csv(pointer.path_wide_dates(), low_memory=False)
        bucketed = PdFields.DEMOGRAPHIC_BUCKET in wide_dates_df.columns
        if bucketed:
            wide_dates_df = wide_dates_df.set_index(
                [CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET]
            )
        else:
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

        dataset = MultiRegionDataset.from_timeseries_wide_dates_df(
            wide_dates_df, bucketed=bucketed
        ).add_static_values(static_df)
        if tag_df_to_concat:
            tag_df = pd.concat(tag_df_to_concat)
            if not bucketed:
                tag_df_add_all_bucket_in_place(tag_df)
            dataset = dataset.append_tag_df(tag_df)

        return dataset

    @staticmethod
    def from_fips_timeseries_df(ts_df: pd.DataFrame) -> "MultiRegionDataset":
        ts_df = _add_location_id(ts_df)
        _add_state_if_missing(ts_df)
        _add_aggregate_level_if_missing(ts_df)

        return MultiRegionDataset.from_timeseries_df(ts_df)

    @staticmethod
    def new_without_timeseries() -> "MultiRegionDataset":
        return MultiRegionDataset.from_fips_timeseries_df(
            pd.DataFrame([], columns=[CommonFields.FIPS, CommonFields.DATE])
        )

    def __post_init__(self):
        """Checks that attributes of this object meet certain expectations."""
        # These asserts provide runtime-checking and a single place for humans reading the code to
        # check what is expected of the attributes, beyond type.
        _check_timeseries_wide_vars_structure(self.timeseries_bucketed, bucketed=True)

        assert self.static.index.names == [CommonFields.LOCATION_ID]
        assert self.static.index.is_unique
        assert self.static.index.is_monotonic_increasing
        assert self.static.columns.intersection(GEO_DATA_COLUMNS).empty

        assert isinstance(self.tag, pd.Series)
        assert self.tag.index.names == _TAG_INDEX_FIELDS
        # TODO(tom): Work out why is_monotonic_increasing is false (just for index with NaT
        #  and real date values?) after calling sort_index(). It may be related to
        #  https://github.com/pandas-dev/pandas/issues/35992 which is fixed in pandas 1.2.0
        # assert self.tag.index.is_monotonic_increasing
        assert self.tag.name == TagField.CONTENT
        # Check that all tag location_id are in timeseries location_id
        assert (
            self.tag.index.unique(TagField.LOCATION_ID)
            .difference(self.timeseries_bucketed.index.unique(CommonFields.LOCATION_ID))
            .empty
        )

        extra_location_ids = self.location_ids.difference(dataset_utils.get_geo_data().index)
        if not extra_location_ids.empty:
            raise AssertionError(f"Unknown locations:\n{extra_location_ids}")

    def append_regions(self, other: "MultiRegionDataset") -> "MultiRegionDataset":
        common_location_id = self.location_ids.intersection(other.location_ids)
        if not common_location_id.empty:
            raise ValueError("Do not use append_regions with duplicate location_id")
        # TODO(tom): check if rename_axis is needed once we have
        #  https://pandas.pydata.org/docs/whatsnew/v1.2.0.html#index-column-name-preservation-when-aggregating
        timeseries_df = (
            pd.concat([self.timeseries_bucketed, other.timeseries_bucketed])
            .sort_index()
            .rename_axis(columns=PdFields.VARIABLE)
        )
        static_df = pd.concat([self.static, other.static]).sort_index()
        tag = pd.concat([self.tag, other.tag]).sort_index()
        return MultiRegionDataset(timeseries_bucketed=timeseries_df, static=static_df, tag=tag)

    def append_tag_df(self, additional_tag_df: pd.DataFrame) -> "MultiRegionDataset":
        """Returns a new dataset with additional_tag_df appended."""
        if additional_tag_df.empty:
            return self
        assert additional_tag_df.columns.symmetric_difference(_TAG_DF_COLUMNS, sort=False).empty
        # Sort by index fields, and within rows having identical index fields, by content. This
        # makes the order of values in combined_series identical, independent of the order they
        # were appended.
        combined_df = (
            self.tag.reset_index()
            .append(additional_tag_df)
            .sort_values(_TAG_INDEX_FIELDS + [TagField.CONTENT])
        )
        combined_series = combined_df.set_index(_TAG_INDEX_FIELDS)[TagField.CONTENT]
        return dataclasses.replace(self, tag=combined_series)

    def replace_tag_df(self, tag_df: pd.DataFrame) -> "MultiRegionDataset":
        """Returns a new dataset with all tags replaced by those in tag_df"""
        return dataclasses.replace(self, tag=_EMPTY_TAG_SERIES).append_tag_df(tag_df)

    def get_one_region(self, region: Region) -> OneRegionTimeseriesDataset:
        try:
            ts_df = self.timeseries.xs(
                region.location_id, level=CommonFields.LOCATION_ID, drop_level=False
            ).reset_index()
        except KeyError:
            ts_df = pd.DataFrame([], columns=[CommonFields.LOCATION_ID, CommonFields.DATE])
        latest_dict = self._location_id_latest_dict(region.location_id)
        bucketed_latest = self._bucketed_latest_for_location_id(region.location_id)
        if ts_df.empty and not latest_dict:
            raise RegionLatestNotFound(region)

        tag = self.tag.loc[[region.location_id]].reset_index(TagField.LOCATION_ID, drop=True)

        return OneRegionTimeseriesDataset(
            region=region, data=ts_df, latest=latest_dict, tag=tag, bucketed_latest=bucketed_latest
        )

    def _location_id_latest_dict(self, location_id: str) -> dict:
        """Returns the latest values dict of a location_id."""
        try:
            attributes_series = self.static_and_timeseries_latest_with_fips().loc[location_id, :]
        except KeyError:
            attributes_series = pd.Series([], dtype=object)
        return attributes_series.where(pd.notnull(attributes_series), None).to_dict()

    def _bucketed_latest_for_location_id(self, location_id: str) -> pd.DataFrame:
        try:
            data = self._timeseries_bucketed_latest_values.loc[location_id, :]
            return data
        except KeyError:
            return pd.DataFrame(
                [],
                index=pd.MultiIndex.from_tuples([], names=[PdFields.DEMOGRAPHIC_BUCKET]),
                columns=self.timeseries_bucketed.columns,
                dtype="float",
            )

    def _location_ids_in_mask(self, region_mask: pipeline.RegionMask) -> pd.Index:
        geo_data = self.geo_data
        rows_key = dataset_utils.make_rows_key(
            geo_data, aggregation_level=region_mask.level, states=region_mask.states,
        )
        return geo_data.loc[rows_key, :].index

    def _regionmaskorregions_to_location_id(
        self, regions_and_masks: Collection[RegionMaskOrRegion]
    ):
        location_ids = set()
        for region_or_mask in regions_and_masks:
            if isinstance(region_or_mask, Region):
                location_ids.add(region_or_mask.location_id)
            else:
                assert isinstance(region_or_mask, pipeline.RegionMask)
                location_ids.update(self._location_ids_in_mask(region_or_mask))
        return pd.Index(sorted(location_ids), dtype=str)

    def get_regions_subset(self, regions: Collection[RegionMaskOrRegion]) -> "MultiRegionDataset":
        location_ids = self._regionmaskorregions_to_location_id(regions)
        return self.get_locations_subset(location_ids)

    def get_locations_subset(self, location_ids: Collection[str]) -> "MultiRegionDataset":
        # If these mask+loc operations show up as a performance issue try changing to xs.
        timeseries_mask = self.timeseries_bucketed.index.get_level_values(
            CommonFields.LOCATION_ID
        ).isin(location_ids)
        timeseries_df = self.timeseries_bucketed.loc[timeseries_mask, :]
        static_mask = self.static.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        static_df = self.static.loc[static_mask, :]
        tag_mask = self.tag.index.get_level_values(CommonFields.LOCATION_ID).isin(location_ids)
        tag = self.tag.loc[tag_mask, :]
        return MultiRegionDataset(timeseries_bucketed=timeseries_df, static=static_df, tag=tag)

    def remove_regions(self, regions: Collection[RegionMaskOrRegion]) -> "MultiRegionDataset":
        location_ids = self._regionmaskorregions_to_location_id(regions)
        return self.remove_locations(location_ids)

    def remove_locations(self, location_ids: Collection[str]) -> "MultiRegionDataset":
        timeseries_mask = self.timeseries_bucketed.index.get_level_values(
            CommonFields.LOCATION_ID
        ).isin(location_ids)
        timeseries_df = self.timeseries_bucketed.loc[~timeseries_mask, :]
        static_mask = self.static.index.get_level_values(CommonFields.LOCATION_ID).isin(
            location_ids
        )
        static_df = self.static.loc[~static_mask, :]
        tag_mask = self.tag.index.get_level_values(CommonFields.LOCATION_ID).isin(location_ids)
        tag = self.tag.loc[~tag_mask, :]
        return MultiRegionDataset(timeseries_bucketed=timeseries_df, static=static_df, tag=tag)

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
            self.geo_data,
            aggregation_level=aggregation_level,
            fips=fips,
            state=state,
            states=states,
            location_id_matches=location_id_matches,
            exclude_county_999=exclude_county_999,
        )
        location_ids = self.geo_data.loc[rows_key, :].index
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
        return dataclasses.replace(
            self, timeseries=self.timeseries.loc[ts_rows_mask, :], timeseries_bucketed=None
        )

    def groupby_region(self) -> pandas.core.groupby.generic.DataFrameGroupBy:
        return self.timeseries.groupby(CommonFields.LOCATION_ID)

    def timeseries_rows(self) -> pd.DataFrame:
        """Returns a DataFrame containing timeseries values and tag_all_bucket, suitable for writing
        to a CSV."""
        # Make a copy to avoid modifying the cached DataFrame
        wide_dates = self.timeseries_bucketed_wide_dates.copy()
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
        ts = self.timeseries_bucketed_wide_dates
        recent_columns_mask = ts.columns >= cutoff_date
        recent_rows_mask = ts.loc[:, recent_columns_mask].notna().any(axis=1)
        timeseries_wide_dates = ts.loc[recent_rows_mask, :]

        # Change DataFrame with date columns to DataFrame with variable columns, similar
        # to a line in from_timeseries_wide_dates_df.
        timeseries_wide_variables = (
            timeseries_wide_dates.stack()
            .unstack(PdFields.VARIABLE)
            .reindex(columns=self.timeseries_bucketed.columns)
            .sort_index()
        )
        # Only keep tag information for timeseries in the new timeseries_wide_dates.
        tag = _slice_with_labels(self.tag, timeseries_wide_dates.index)
        return dataclasses.replace(self, timeseries_bucketed=timeseries_wide_variables, tag=tag,)

    def to_csv(self, path: pathlib.Path, include_latest=True):
        """Persists timeseries to CSV.

        Args:
            path: Path to write to.
        """
        timeseries_data = self.timeseries.reset_index()
        _add_fips_if_missing(timeseries_data)

        if include_latest:
            latest_data = self.static.reset_index()
            _add_fips_if_missing(latest_data)
        else:
            latest_data = pd.DataFrame([])

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

        # The values we write are generally ratios (such as test positivity) where we only need ~5
        # digits beyond the decimal point or integers that we'd like to preserve as an exact value.
        # I'm concerned that a delta calculated from a rounded cumulative count may differ
        # significantly from a delta calculated from an unrounded version of the same count. Also
        # a timeseries delta calculated from a rounded cumulative count may include artifacts
        # that look like the step function. %.9g preserves the exact value for US vaccine counts
        # that are now over 100M. Unfortunately this also writes unnecessarily high precision for
        # floats but I don't see an easy solution with to_csv float_format.
        # https://trello.com/c/aDGn57Df/1192-change-combined-data-from-csv-to-parquet will remove
        # the need to format values as strings.
        csv_buf = wide_df.to_csv(index=True, float_format="%.9g")
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
        timeseries_df = self.timeseries_bucketed.drop(columns, axis="columns", errors="ignore")
        static_df = self.static.drop(columns, axis="columns", errors="ignore")
        tag = self.tag[~self.tag.index.get_level_values(PdFields.VARIABLE).isin(columns)]
        return MultiRegionDataset(timeseries_bucketed=timeseries_df, static=static_df, tag=tag)

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
        common_ts_columns = set(other.timeseries_bucketed.columns) & set(
            self.timeseries_bucketed.columns
        )
        if common_ts_columns:
            # columns to be joined need to be disjoint
            raise ValueError(f"Columns are in both dataset: {common_ts_columns}")
        combined_df = pd.concat([self.timeseries_bucketed, other.timeseries_bucketed], axis=1)
        combined_tag = pd.concat([self.tag, other.tag]).sort_index()
        return MultiRegionDataset(
            timeseries_bucketed=combined_df, static=self.static, tag=combined_tag
        )

    def iter_one_regions(self) -> Iterable[Tuple[Region, OneRegionTimeseriesDataset]]:
        """Iterates through all the regions in this object"""
        for location_id, timeseries_group in self.timeseries.groupby(
            CommonFields.LOCATION_ID, as_index=True
        ):
            latest_dict = self._location_id_latest_dict(location_id)
            region = Region.from_location_id(location_id)
            tag = self.tag.loc[[region.location_id]].reset_index(TagField.LOCATION_ID, drop=True)
            bucketed_latest = self._bucketed_latest_for_location_id(location_id)

            yield region, OneRegionTimeseriesDataset(
                region,
                timeseries_group.reset_index(),
                latest_dict,
                tag=tag,
                bucketed_latest=bucketed_latest,
            )


def _remove_padded_nans(df, columns):
    if df[columns].isna().all(axis=None):
        return df.loc[[False] * len(df), :].reset_index(drop=True)

    first_valid_index = min(df[column].first_valid_index() for column in columns)
    last_valid_index = max(df[column].last_valid_index() for column in columns)
    df = df.iloc[first_valid_index : last_valid_index + 1]
    return df.reset_index(drop=True)


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


class DatasetName(str):
    """Human readable name for a dataset. In the future this may be an enum, for now it
    provides some type safety."""

    pass


def _slice_with_labels(series: pd.Series, labels: pd.MultiIndex) -> pd.Series:
    """Emulates what I'd like `series.xs(labels, level=labels.names, drop=False)` to do
    and also doesn't raise a KeyError."""

    # Somewhat inspired by https://stackoverflow.com/questions/42733118/how-to-select-a-subset-from

    # Change the input series to have index labels in same order as labels
    reindexed = series.reset_index().set_index(labels.names)
    # Select the rows (avoiding KeyError of reindexed.loc[labels]) then restore the index to match
    # the input Series
    return (
        reindexed.loc[reindexed.index.isin(labels)]
        .reset_index()
        .set_index(series.index.names)[series.name]
    )


def combined_datasets(
    timeseries_field_datasets: Mapping[FieldName, List[MultiRegionDataset]],
    static_field_datasets: Mapping[FieldName, List[MultiRegionDataset]],
) -> MultiRegionDataset:
    """Creates a dataset that gets each field from a list of datasets.

    For each region, the timeseries from the first dataset in the list with a real value is returned.
    """
    if timeseries_field_datasets:
        # First make a map from dataset to table with bool values that represent what timeseries
        # (or distribution) has any real value in that dataset.
        # In the table the index labels are currently just location_id (but could be expanded to
        # include the distribution such as 'all', 'age', 'race', ... if these are combined
        # separately) and columns are fields.
        all_timeseries_datasets = set(chain.from_iterable(timeseries_field_datasets.values()))
        datasets_wide = _datasets_wide_var_not_null(all_timeseries_datasets)
        # Then make a map from dataset to table with the same index and columns but only True
        # where that particular data will be copied to the output. These tables will have a
        # subset of the True values in `datasets_wide`.
        datasets_output = _pick_first_with_field(datasets_wide, timeseries_field_datasets)
        # Finally copy the timeseries and tags that were selected from each dataset.
        ts_bucketed, tags = _combine_timeseries(datasets_output)
    else:
        ts_bucketed, tags = _combine_timeseries({})

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
    if static_series:
        output_static_df = pd.concat(
            static_series, axis=1, sort=True, verify_integrity=True
        ).rename_axis(index=CommonFields.LOCATION_ID)
    else:
        output_static_df = EMPTY_STATIC_DF

    return MultiRegionDataset(timeseries_bucketed=ts_bucketed, tag=tags, static=output_static_df,)


def _pick_first_with_field(
    datasets_wide: Mapping[MultiRegionDataset, pd.DataFrame],
    timeseries_field_datasets: Mapping[FieldName, List[MultiRegionDataset]],
) -> Mapping[MultiRegionDataset, pd.DataFrame]:
    """Creates a DataFrame for each dataset that has True for the subset of datasets_wide that is
    selected according to timeseries_field_datasets"""
    common_index = more_itertools.first(datasets_wide.values()).index
    for df in datasets_wide.values():
        assert df.columns.names == [PdFields.VARIABLE]
        assert df.index.equals(common_index)
    assert common_index.names == [CommonFields.LOCATION_ID]
    datasets_output = {
        ds: pd.DataFrame([], index=common_index, dtype="bool").rename_axis(
            columns=PdFields.VARIABLE
        )
        for ds in datasets_wide.keys()
    }
    for field, datasets in timeseries_field_datasets.items():
        location_id_so_far = pd.Series([], dtype=bool).reindex(index=common_index, fill_value=False)
        for dataset in datasets:
            try:
                this_dataset = datasets_wide[dataset].loc[:, field]
            except KeyError:
                continue
            datasets_output[dataset].loc[:, field] = this_dataset & ~location_id_so_far
            location_id_so_far = location_id_so_far | this_dataset
    return datasets_output


def _datasets_wide_var_not_null(
    datasets: Collection[MultiRegionDataset],
) -> Mapping[MultiRegionDataset, pd.DataFrame]:
    """Makes a map from dataset to a DataFrame with a True where the timeseries has a real value."""
    datasets_wide = {ds: ds.wide_var_not_null for ds in datasets}
    return _with_common_index(datasets_wide)


def _with_common_index(
    datasets_wide: Mapping[MultiRegionDataset, pd.DataFrame]
) -> Mapping[MultiRegionDataset, pd.DataFrame]:
    dataset_wide_first = more_itertools.first(datasets_wide.values())
    assert dataset_wide_first.index.names == [CommonFields.LOCATION_ID]
    assert dataset_wide_first.columns.names == [PdFields.VARIABLE]
    common_index = pd.Index(
        np.unique(np.concatenate([ds_w.index for ds_w in datasets_wide.values()])),
        name=dataset_wide_first.index.name,
    )
    datasets_wide = {
        ds: ds_w.reindex(index=common_index, fill_value=False) for ds, ds_w in datasets_wide.items()
    }
    return datasets_wide


def _combine_timeseries(
    datasets_output: Mapping[MultiRegionDataset, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Combine timeseries selected from each dataset into one structure.

    Args:
        datasets_output: Map from dataset to a DataFrame of which timeseries to output
    Returns:
        Tuple of timeseries_bucketed and tag, suitable for MultiRegionDataset.__init__
    """
    ts_bucketed_long_to_concat = []
    tags_to_concat = []
    for ds, outputs in datasets_output.items():
        if outputs.empty:
            continue
        # Change False to nan so they are dropped when stacking.
        output_labels = outputs.replace({False: np.nan}).stack().index
        if output_labels.empty:
            continue
        assert output_labels.names == [CommonFields.LOCATION_ID, PdFields.VARIABLE]
        ts_bucketed_long_to_concat.append(
            _slice_with_labels(ds.timeseries_bucketed_long, output_labels)
        )
        tags_to_concat.append(_slice_with_labels(ds.tag, output_labels))

    if ts_bucketed_long_to_concat:
        ts_bucketed = pd.concat(ts_bucketed_long_to_concat).unstack(PdFields.VARIABLE).sort_index()
        tags = pd.concat(tags_to_concat).sort_index()
    else:
        ts_bucketed = EMPTY_TIMESERIES_BUCKETED_WIDE_VARIABLES_DF
        tags = _EMPTY_TAG_SERIES

    return ts_bucketed, tags


def make_source_tags(ds_in: MultiRegionDataset) -> MultiRegionDataset:
    """Convert provenance and source_url tags into source tags."""
    # TODO(tom): Make sure taglib.Source.rename_and_make_tag_df is tested well without tests that
    #  call this function, then delete this function.
    # Separate ds_in.tag into tags to transform into `source` tags and tags to copy unmodified.
    ds_in_tag_extract_mask = ds_in.tag.index.get_level_values(TagField.TYPE).isin(
        [TagType.PROVENANCE, TagType.SOURCE_URL]
    )
    other_tags_df = ds_in.tag.loc[~ds_in_tag_extract_mask].reset_index()
    # Fill in missing elements of the DataFrame with None in two steps because it can't be done
    # by `unstack`.
    extracted_tags_df = (
        ds_in.tag.loc[ds_in_tag_extract_mask]
        .unstack(TagField.TYPE, fill_value=pd.NA)
        .replace(
            {pd.NA: None}
        )  # From https://github.com/pandas-dev/pandas/issues/17494#issuecomment-328966324
    )

    source_df = taglib.Source.rename_and_make_tag_df(
        extracted_tags_df, rename={TagType.PROVENANCE: "type", TagType.SOURCE_URL: "url"}
    )

    return ds_in.replace_tag_df(pd.concat([source_df, other_tags_df]))


def make_source_url_tags(ds_in: MultiRegionDataset) -> MultiRegionDataset:
    """Make source_url tags from source tags"""
    # TODO(tom): When we have clearer view of how we want to build materialized views of tags
    #  replace use of this function.
    assert TagType.SOURCE_URL not in ds_in.tag_all_bucket.index.unique(TagField.TYPE)
    try:
        source_tags = ds_in.tag.xs(TagType.SOURCE, level=TagField.TYPE)
    except KeyError:
        return ds_in
    source_url = (
        source_tags.apply(lambda content: taglib.Source.make_instance(content=content).url)
        .dropna()
        .rename(TagField.CONTENT)
        .reset_index()
    )
    source_url[TagField.TYPE] = TagType.SOURCE_URL
    return ds_in.append_tag_df(source_url)
