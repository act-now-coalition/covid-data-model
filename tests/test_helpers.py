import abc
import dataclasses
import enum
import functools
import inspect
import io
from typing import Collection
from typing import Iterable
from typing import List

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
from datapublic.common_fields import CommonFields
from datapublic.common_fields import DemographicBucket
from datapublic.common_fields import FieldName
from datapublic.common_fields import PdFields

from libs.dataclass_utils import dataclass_with_default_init
from libs.datasets import combined_datasets
from libs.datasets import dataset_utils
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets.taglib import TagField
from libs.datasets.taglib import TagType
from libs.datasets.taglib import UrlStr
from libs.datasets.timeseries import MultiRegionDataset
from libs.pipeline import Region


# This is a totally bogus fips/region/location that we've been using as a default in some test
# cases. It is factored out here in an attempt to reduce how much it is hard-coded into our source.
DEFAULT_FIPS = "97222"
DEFAULT_REGION = Region.from_fips(DEFAULT_FIPS)

DEFAULT_START_DATE = "2020-04-01"


@dataclass_with_default_init(frozen=True)
class TimeseriesLiteral:
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
        self.__default_init__(  # pylint: disable=E1101
            *args,
            provenance=combined_datasets.to_list(provenance),
            source_url=combined_datasets.to_list(source_url),
            source=combined_datasets.to_list(source),
            **kwargs,
        )


def series_with_date_index(data, date: str = "2020-08-25", **series_kwargs):
    """Creates a series with an increasing date index."""
    date_series = pd.date_range(date, periods=len(data), freq="D")
    return pd.Series(data, index=date_series, **series_kwargs)


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
    tag_type: TagType = TagType.CUMULATIVE_TAIL_TRUNCATED, **kwargs
) -> taglib.TagInTimeseries:
    if tag_type in timeseries.ANNOTATION_TAG_TYPES:
        # Force to the expected types and add defaults if not in kwargs
        kwargs["original_observation"] = float(kwargs.get("original_observation", 10))
        kwargs["date"] = pd.to_datetime(kwargs.get("date", "2020-04-02"))
    elif tag_type is taglib.TagType.KNOWN_ISSUE:
        kwargs["date"] = pd.to_datetime(kwargs.get("date", "2020-04-02")).date()
    elif tag_type is taglib.TagType.DROP_FUTURE_OBSERVATION:
        # make_tag does not have a default `after`; force tests to provide it explicitly.
        kwargs["after"] = pd.to_datetime(kwargs["after"]).date()

    return taglib.TAG_TYPE_TO_CLASS[tag_type](**kwargs)


def flatten_3_nested_dict(
    nested: Mapping[Any, Mapping[Any, Mapping[Any, Any]]], index_names: Sequence[str]
) -> pd.Series:
    """Returns a Series with MultiIndex created from the keys of a nested dictionary."""
    # I was attempting to use this in build_dataset but was foiled by the access to
    # region.location_id when making the MultiIndex.
    dict_tuple_to_value = {
        (key1, key2, key3): value
        for key1 in nested.keys()
        for key2 in nested[key1].keys()
        for key3, value in nested[key1][key2].items()
    }
    index = pd.MultiIndex.from_tuples(dict_tuple_to_value.keys(), names=index_names)
    return pd.Series(list(dict_tuple_to_value.values()), index=index)


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

    tags_to_concat = []
    region_var_bucket_timeseries = {}
    for region, region_metrics in metrics_by_region_then_field_name.items():
        for var_name, var_buckets in region_metrics.items():
            for bucket_name, ts_literal in iter_buckets(var_buckets):
                if isinstance(ts_literal, TimeseriesLiteral):
                    timeseries_data = ts_literal.data
                    records = list(ts_literal.annotation)
                    records.extend(ts_literal.source)
                    records.extend(
                        make_tag(TagType.PROVENANCE, source=provenance)
                        for provenance in ts_literal.provenance
                    )
                    records.extend(
                        make_tag(TagType.SOURCE_URL, source=url) for url in ts_literal.source_url
                    )
                    tags_to_concat.append(make_tag_df(region, var_name, bucket_name, records))
                else:
                    timeseries_data = ts_literal
                region_var_bucket_timeseries[(region, var_name, bucket_name)] = timeseries_data

    if region_var_bucket_timeseries:
        # Find the longest sequence in region_var_bucket_timeseries.values(). Make a DatetimeIndex
        # with that many dates.
        sequence_lengths = max(len(seq) for seq in region_var_bucket_timeseries.values())
        dates = pd.date_range(
            start_date, periods=sequence_lengths, freq="D", name=CommonFields.DATE
        )

        index = pd.MultiIndex.from_tuples(
            [
                (region.location_id, var, bucket)
                for region, var, bucket in region_var_bucket_timeseries.keys()
            ],
            names=[CommonFields.LOCATION_ID, PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET],
        )

        df = pd.DataFrame(list(region_var_bucket_timeseries.values()), index=index, columns=dates)
        df = df.fillna(np.nan).apply(pd.to_numeric)

        dataset = timeseries.MultiRegionDataset.from_timeseries_wide_dates_df(df, bucketed=True)
    else:
        dataset = timeseries.MultiRegionDataset.new_without_timeseries()

    if timeseries_columns:
        new_timeseries = _add_missing_columns(dataset.timeseries, timeseries_columns)
        dataset = dataclasses.replace(dataset, timeseries=new_timeseries, timeseries_bucketed=None)

    if tags_to_concat:
        dataset = dataset.append_tag_df(pd.concat(tags_to_concat))

    if static_by_region_then_field_name:
        static_df_rows = []
        for region, data in static_by_region_then_field_name.items():
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
    compare_tags=True,
):
    """Asserts that two datasets contain similar date, ignoring order."""
    ts1 = _timeseries_sorted_by_location_date(
        ds1, drop_na=drop_na_timeseries, drop_na_dates=drop_na_dates
    )
    ts2 = _timeseries_sorted_by_location_date(
        ds2, drop_na=drop_na_timeseries, drop_na_dates=drop_na_dates
    )
    pd.testing.assert_frame_equal(ts1, ts2, check_like=True, check_dtype=False)
    latest1 = _latest_sorted_by_location_date(ds1, drop_na_latest)
    latest2 = _latest_sorted_by_location_date(ds2, drop_na_latest)
    pd.testing.assert_frame_equal(latest1, latest2, check_like=True, check_dtype=False)
    # Somehow tests/libs/datasets/combined_dataset_utils_test.py::test_update_and_load has
    # two provenance Series that are empty but assert_series_equal fails with message
    # 'Attribute "inferred_type" are different'. Don't call it when both series are empty.
    if not (ds1.provenance.empty and ds2.provenance.empty):
        pd.testing.assert_series_equal(ds1.provenance, ds2.provenance)

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


def get_concrete_subclasses_not_in_tests(cls) -> Iterable[Type]:
    """Yields all subclasses of `cls` that have no abstract methods, do not directly subclass
    abc.ABC and are not in a `tests` module."""
    for subcls in get_subclasses(cls):
        if (
            not inspect.isabstract(subcls)
            and abc.ABC not in subcls.__bases__
            and not subcls.__module__.startswith("tests.")
        ):
            yield subcls


def read_csv_str(
    csv: str,
    *,
    skip_spaces: bool = False,
    parse_dates: Optional[List] = None,
    dtype: Optional[Mapping] = None,
) -> pd.DataFrame:
    """Reads a CSV passed as a string.

    Args:
        csv: String content to parse as a CSV
        skip_spaces: If True, removes all " " from csv
        parse_dates: Passed to pd.read_csv. If None and 'date' is in the header then ['date'].
        dtype: Passed to pd.read_csv. If None and 'fips' is in the header it is parsed as a str.
    """
    if skip_spaces:
        csv = csv.replace(" ", "")
    header = more_itertools.first(csv.splitlines()).split(",")
    if parse_dates is None:
        if CommonFields.DATE in header:
            parse_dates = [CommonFields.DATE]
        else:
            parse_dates = []
    if dtype is None:
        if CommonFields.FIPS in header:
            dtype = {CommonFields.FIPS: str}
        else:
            dtype = {}

    return pd.read_csv(io.StringIO(csv), parse_dates=parse_dates, dtype=dtype)


@functools.lru_cache(None)
def load_test_dataset() -> MultiRegionDataset:
    return MultiRegionDataset.from_wide_dates_csv(
        dataset_utils.TEST_COMBINED_WIDE_DATES_CSV_PATH
    ).add_static_csv_file(dataset_utils.TEST_COMBINED_STATIC_CSV_PATH)


def assert_enum_names_match_values(enum_cls: Type[enum.Enum], exceptions: Collection = ()):
    mismatches = []
    for val in enum_cls:
        if val in exceptions:
            continue
        if val.name.lower() != val.value:
            mismatches.append(val)
    if mismatches:
        suggestion = "\n".join(f"    {v.name} = {repr(v.name.lower())}" for v in mismatches)
        print(f"fix for enum name and value mismatches:\n{suggestion}")
    assert mismatches == []
