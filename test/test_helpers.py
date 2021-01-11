import dataclasses
from collections import UserList
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import more_itertools
import pandas as pd
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import timeseries
from libs.datasets.timeseries import TagField
from libs.pipeline import Region


# This is a totally bogus fips/region/location that we've been using as a default in some test
# cases. It is factored out here in an attempt to reduce how much it is hard-coded into our source.
DEFAULT_FIPS = "97222"
DEFAULT_REGION = Region.from_fips(DEFAULT_FIPS)


class TimeseriesLiteral(UserList):
    """Represents a timeseries literal, a sequence of floats and provenance string."""

    def __init__(
        self,
        ts_list,
        *,
        provenance: Union[str, List[str]] = "",
        annotation: Sequence[timeseries.TagInTimeseries] = (),
    ):
        super().__init__(ts_list)
        self.provenance = provenance
        self.annotation = annotation


def make_tag_df(
    region: Region, metric: CommonFields, records: List[timeseries.TagInTimeseries]
) -> pd.DataFrame:
    df = pd.DataFrame.from_records(
        [dataclasses.astuple(r) for r in records],
        columns=[TagField.TYPE, TagField.DATE, TagField.CONTENT],
    )
    df[TagField.DATE] = pd.to_datetime(df[TagField.DATE])
    df[TagField.LOCATION_ID] = region.location_id
    df[TagField.VARIABLE] = metric
    return df


def make_tag(
    type: TagType = TagType.CUMULATIVE_TAIL_TRUNCATED,
    date: Union[pd.Timestamp, str] = "2020-04-02",
    content: str = "taggy",
) -> timeseries.TagInTimeseries:
    return timeseries.TagInTimeseries(type, pd.to_datetime(date), content)


def build_dataset(
    metrics_by_region_then_field_name: Mapping[
        Region, Mapping[FieldName, Union[Sequence[float], TimeseriesLiteral]]
    ],
    *,
    start_date="2020-04-01",
    timeseries_columns: Optional[Sequence[FieldName]] = None,
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
    region_var_seq = {
        (region, variable): metrics_by_region_then_field_name[region][variable]
        for region in metrics_by_region_then_field_name.keys()
        for variable in metrics_by_region_then_field_name[region].keys()
    }

    # Make sure there is only one len among all of region_var_seq.values(). Make a DatetimeIndex
    # with that many dates.
    sequence_lengths = more_itertools.one(set(len(seq) for seq in region_var_seq.values()))
    dates = pd.date_range(start_date, periods=sequence_lengths, freq="D", name=CommonFields.DATE)

    index = pd.MultiIndex.from_tuples(
        [(region.location_id, var) for region, var in region_var_seq.keys()],
        names=[CommonFields.LOCATION_ID, PdFields.VARIABLE],
    )

    df = pd.DataFrame(list(region_var_seq.values()), index=index, columns=dates)
    df = df.fillna(np.nan).apply(pd.to_numeric)

    dataset = timeseries.MultiRegionDataset.from_timeseries_wide_dates_df(df)

    if timeseries_columns:
        new_timeseries = _add_missing_columns(dataset.timeseries, timeseries_columns)
        dataset = dataclasses.replace(dataset, timeseries=new_timeseries)

    tags_to_concat = []
    for (region, var), ts_literal in region_var_seq.items():
        if not isinstance(ts_literal, TimeseriesLiteral):
            continue

        records = list(ts_literal.annotation)
        if not ts_literal.provenance:
            provenance_list = []
        elif isinstance(ts_literal.provenance, str):
            provenance_list = [ts_literal.provenance]
        else:
            provenance_list = ts_literal.provenance
        records.extend(
            timeseries.TagInTimeseries(timeseries.TagType.PROVENANCE, pd.NaT, provenance)
            for provenance in provenance_list
        )
        tags_to_concat.append(make_tag_df(region, var, records))

    if tags_to_concat:
        dataset = dataset.append_tag_df(pd.concat(tags_to_concat))

    return dataset


def build_default_region_dataset(
    metrics: Mapping[FieldName, Union[Sequence[float], TimeseriesLiteral]],
    *,
    region=DEFAULT_REGION,
) -> timeseries.MultiRegionDataset:
    """Returns a `MultiRegionDataset` containing metrics in one region"""
    return build_dataset({region: metrics})


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
    """Returns the timeseries data, sorted by LOCATION_ID and DATE."""
    df = dataset.timeseries
    if drop_na:
        df = df.dropna("columns", "all")
    if drop_na_dates:
        df = df.dropna("rows", "all")
    df = df.reset_index().sort_values(
        [CommonFields.LOCATION_ID, CommonFields.DATE], ignore_index=True
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
    # Somehow test/libs/datasets/combined_dataset_utils_test.py::test_update_and_load has
    # two provenance Series that are empty but assert_series_equal fails with message
    # 'Attribute "inferred_type" are different'. Don't call it when both series are empty.
    if not (ds1.provenance.empty and ds2.provenance.empty):
        pd.testing.assert_series_equal(
            ds1.provenance, ds2.provenance, check_less_precise=check_less_precise
        )

    if compare_tags:
        tag1 = ds1.tag.astype("string")
        tag2 = ds2.tag.astype("string")
        pd.testing.assert_series_equal(tag1, tag2)
