import abc
import dataclasses
import inspect
from typing import Iterable
from typing import List

from collections import UserList
from typing import Any
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import more_itertools
import pandas as pd
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields

from libs.dataclass_utils import dataclass_with_default_init
from libs.datasets import combined_datasets
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets.taglib import TagField
from libs.datasets.taglib import TagType
from libs.datasets.taglib import UrlStr
from libs.pipeline import Region


# This is a totally bogus fips/region/location that we've been using as a default in some test
# cases. It is factored out here in an attempt to reduce how much it is hard-coded into our source.
DEFAULT_FIPS = "97222"
DEFAULT_REGION = Region.from_fips(DEFAULT_FIPS)

DEFAULT_START_DATE = "2020-04-01"


@dataclass_with_default_init(frozen=True)
class TimeseriesLiteral(UserList):
    """Represents a timeseries literal: a sequence of floats and some related attributes."""

    data: Sequence[float]
    provenance: Sequence[str]
    source_url: Sequence[UrlStr]
    source: Sequence[taglib.Source] = ()
    annotation: Sequence[taglib.TagInTimeseries] = ()

    # noinspection PyMissingConstructor
    def __init__(
        self,
        *args,
        provenance: Union[None, str, List[str]] = None,
        source_url: Union[None, UrlStr, List[UrlStr]] = None,
        source: Union[None, taglib.Source, List[taglib.Source]] = None,
        **kwargs,
    ):
        """Initialize `self`, doing some type conversion."""
        # UserList.__init__ attempts to set self.data, which fails on this frozen class. Instead
        # let the dataclasses code initialize `data`.
        self.__default_init__(  # pylint: disable=E1101
            *args,
            provenance=combined_datasets.to_list(provenance),
            source_url=combined_datasets.to_list(source_url),
            source=combined_datasets.to_list(source),
            **kwargs,
        )


def make_tag_df(
    region: Region,
    metric: CommonFields,
    bucket: DemographicBucket,
    records: List[taglib.TagInTimeseries],
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            TagField.TYPE: [r.tag_type for r in records],
            TagField.CONTENT: [r.content for r in records],
        }
    )
    df[TagField.LOCATION_ID] = region.location_id
    df[TagField.VARIABLE] = metric
    df[TagField.DEMOGRAPHIC_BUCKET] = bucket
    return df


def make_tag(
    tag_type: TagType = TagType.CUMULATIVE_TAIL_TRUNCATED, **kwargs,
) -> taglib.TagInTimeseries:
    if tag_type in timeseries.ANNOTATION_TAG_TYPES:
        # Force to the expected types and add defaults if not in kwargs
        kwargs["original_observation"] = float(kwargs.get("original_observation", 10))
        kwargs["date"] = pd.to_datetime(kwargs.get("date", "2020-04-02"))

    return taglib.TAG_TYPE_TO_CLASS[tag_type](**kwargs)


SingleOrBucketedTimeseriesLiteral = NewType(
    "SingleOrBucketedTimeseriesLiteral",
    Union[
        Sequence[float],
        TimeseriesLiteral,
        Mapping[DemographicBucket, Union[Sequence[float], TimeseriesLiteral]],
    ],
)


def build_dataset(
    metrics_by_region_then_field_name: Mapping[
        Region, Mapping[FieldName, SingleOrBucketedTimeseriesLiteral]
    ],
    *,
    start_date=DEFAULT_START_DATE,
    timeseries_columns: Optional[Sequence[FieldName]] = None,
    static_by_region_then_field_name: Optional[Mapping[Region, Mapping[FieldName, Any]]] = None,
) -> timeseries.MultiRegionDataset:
    """Returns a dataset for multiple regions and metrics.
    Args:
        metrics_by_region_then_field_name: Each sequence of values and TimeseriesLiteral must have
            at least one real value and identical length. The oldest date is the 0th element.
        start_date: The oldest date of each timeseries.
        timeseries_columns: Columns that will exist in the returned dataset, even if all NA
    """
    # From https://stackoverflow.com/a/47416248. Make a dictionary listing all the timeseries
    # sequences in metrics.
    def iter_buckets(
        buckets: SingleOrBucketedTimeseriesLiteral,
    ) -> Tuple[DemographicBucket, Union[Sequence[float], TimeseriesLiteral]]:
        if isinstance(buckets, Mapping):
            yield from buckets.items()
        else:
            yield DemographicBucket("all"), buckets

    region_var_bucket_seq = {
        (region, var_name, bucket_name): bucket_ts
        for region in metrics_by_region_then_field_name.keys()
        for var_name, var_buckets in metrics_by_region_then_field_name[region].items()
        for bucket_name, bucket_ts in iter_buckets(var_buckets)
    }

    # Make sure there is only one len among all of region_var_bucket_seq.values(). Make a DatetimeIndex
    # with that many dates.
    sequence_lengths = more_itertools.one(set(len(seq) for seq in region_var_bucket_seq.values()))
    dates = pd.date_range(start_date, periods=sequence_lengths, freq="D", name=CommonFields.DATE)

    index = pd.MultiIndex.from_tuples(
        [(region.location_id, var, bucket) for region, var, bucket in region_var_bucket_seq.keys()],
        names=[CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET],
    )

    df = pd.DataFrame(list(region_var_bucket_seq.values()), index=index, columns=dates)
    df = df.fillna(np.nan).apply(pd.to_numeric)

    dataset = timeseries.MultiRegionDataset.from_timeseries_wide_dates_df(df, bucketed=True)

    if timeseries_columns:
        new_timeseries = _add_missing_columns(dataset.timeseries, timeseries_columns)
        dataset = dataclasses.replace(dataset, timeseries=new_timeseries, timeseries_bucketed=None)

    tags_to_concat = []
    for (region, var, bucket), ts_literal in region_var_bucket_seq.items():
        if not isinstance(ts_literal, TimeseriesLiteral):
            continue

        records = list(ts_literal.annotation)
        records.extend(ts_literal.source)
        records.extend(
            make_tag(TagType.PROVENANCE, source=provenance) for provenance in ts_literal.provenance
        )
        records.extend(make_tag(TagType.SOURCE_URL, source=url) for url in ts_literal.source_url)
        tags_to_concat.append(make_tag_df(region, var, bucket, records))

    if tags_to_concat:
        dataset = dataset.append_tag_df(pd.concat(tags_to_concat))

    if static_by_region_then_field_name:
        static_df_rows = []
        for region, data in static_by_region_then_field_name.items():
            assert CommonFields.AGGREGATE_LEVEL not in data
            assert CommonFields.STATE not in data
            assert CommonFields.COUNTY not in data
            row = {
                CommonFields.LOCATION_ID: region.location_id,
                **data,
            }
            static_df_rows.append(row)

        attributes_df = pd.DataFrame(static_df_rows)
        dataset = dataset.add_static_values(attributes_df)

    return dataset


def build_default_region_dataset(
    metrics: Mapping[FieldName, SingleOrBucketedTimeseriesLiteral],
    *,
    region=DEFAULT_REGION,
    start_date=DEFAULT_START_DATE,
    static: Optional[Mapping[FieldName, Any]] = None,
) -> timeseries.MultiRegionDataset:
    """Returns a `MultiRegionDataset` containing metrics in one region"""
    return build_dataset(
        {region: metrics},
        start_date=start_date,
        static_by_region_then_field_name=({region: static} if static else None),
    )


def build_one_region_dataset(
    metrics: Mapping[FieldName, Sequence[float]],
    *,
    region: Region = DEFAULT_REGION,
    start_date="2020-08-25",
    timeseries_columns: Optional[Sequence[FieldName]] = None,
    latest_override: Optional[Mapping[FieldName, Any]] = None,
) -> timeseries.OneRegionTimeseriesDataset:
    """Returns a `OneRegionTimeseriesDataset` with given timeseries metrics, each having the same
    length.

    Args:
        timeseries_columns: Columns that will exist in the returned dataset, even if all NA
        latest_override: values added to the returned `OneRegionTimeseriesDataset.latest`
    """
    one_region = build_dataset(
        {region: metrics}, start_date=start_date, timeseries_columns=timeseries_columns
    ).get_one_region(region)
    if latest_override:
        new_latest = {**one_region.latest, **latest_override}
        one_region = dataclasses.replace(one_region, latest=new_latest)
    return one_region


def _add_missing_columns(df: pd.DataFrame, timeseries_columns: Sequence[str]):
    """Returns a copy of df with any columns not in timeseries_columns appended."""
    new_columns = [col for col in timeseries_columns if col not in df.columns]
    return df.reindex(columns=[*df.columns, *new_columns])


def _timeseries_sorted_by_location_date(
    dataset: timeseries.MultiRegionDataset, *, drop_na: bool, drop_na_dates: bool
) -> pd.DataFrame:
    """Returns the timeseries data, sorted by LOCATION_ID, DEMOGRAPHIC_BUCKET, DATE."""
    df = dataset.timeseries_bucketed
    if drop_na:
        df = df.dropna("columns", "all")
    if drop_na_dates:
        df = df.dropna("rows", "all")
    df = df.reset_index().sort_values(
        [CommonFields.LOCATION_ID, PdFields.DEMOGRAPHIC_BUCKET, CommonFields.DATE],
        ignore_index=True,
    )
    return df


def _latest_sorted_by_location_date(
    ts: timeseries.MultiRegionDataset, drop_na: bool
) -> pd.DataFrame:
    """Returns the latest data, sorted by LOCATION_ID."""
    df = ts.static_and_timeseries_latest_with_fips().sort_values(
        [CommonFields.LOCATION_ID], ignore_index=True
    )
    if drop_na:
        df = df.dropna("columns", "all")
    return df


def assert_dataset_like(
    ds1: timeseries.MultiRegionDataset,
    ds2: timeseries.MultiRegionDataset,
    *,
    drop_na_timeseries=False,
    drop_na_latest=False,
    drop_na_dates=False,
    check_less_precise=False,
    compare_tags=True,
):
    """Asserts that two datasets contain similar date, ignoring order."""
    ts1 = _timeseries_sorted_by_location_date(
        ds1, drop_na=drop_na_timeseries, drop_na_dates=drop_na_dates
    )
    ts2 = _timeseries_sorted_by_location_date(
        ds2, drop_na=drop_na_timeseries, drop_na_dates=drop_na_dates
    )
    pd.testing.assert_frame_equal(
        ts1, ts2, check_like=True, check_dtype=False, check_less_precise=check_less_precise
    )
    latest1 = _latest_sorted_by_location_date(ds1, drop_na_latest)
    latest2 = _latest_sorted_by_location_date(ds2, drop_na_latest)
    pd.testing.assert_frame_equal(
        latest1, latest2, check_like=True, check_dtype=False, check_less_precise=check_less_precise
    )
    # Somehow tests/libs/datasets/combined_dataset_utils_test.py::test_update_and_load has
    # two provenance Series that are empty but assert_series_equal fails with message
    # 'Attribute "inferred_type" are different'. Don't call it when both series are empty.
    if not (ds1.provenance.empty and ds2.provenance.empty):
        pd.testing.assert_series_equal(
            ds1.provenance, ds2.provenance, check_less_precise=check_less_precise
        )

    if compare_tags:
        tag1 = ds1.tag.astype("string")
        tag2 = ds2.tag.astype("string")
        # Don't check the index types because they don't matter and some tests end up with different
        # types that otherwise compare as equal.
        pd.testing.assert_series_equal(tag1, tag2, check_index_type=False)


def get_subclasses(cls) -> Iterable[Type]:
    """Yields all subclasses of `cls`."""
    # From https://stackoverflow.com/a/33607093
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        yield subclass


def get_concrete_subclasses(cls) -> Iterable[Type]:
    """Yields all subclasses of `cls` that have no abstract methods and do not directly subclass
    abc.ABC."""
    for subcls in get_subclasses(cls):
        if not inspect.isabstract(subcls) and abc.ABC not in subcls.__bases__:
            yield subcls
