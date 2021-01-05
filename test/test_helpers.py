import dataclasses
from collections import UserList
from typing import Any
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
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


# A Tuple of the type, a timestamp and tag content.
AnnotationInTimeseriesLiteral = Tuple[timeseries.TagType, Union[pd.Timestamp, str], str]


class TimeseriesLiteral(UserList):
    """Represents a timeseries literal, a sequence of floats and provenance string."""

    def __init__(
        self,
        ts_list,
        *,
        provenance: str = "",
        annotation: Sequence[AnnotationInTimeseriesLiteral] = (),
    ):
        super().__init__(ts_list)
        self.provenance = provenance
        self.annotation = annotation


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
    loc_var_seq = {
        (region.location_id, variable): metrics_by_region_then_field_name[region][variable]
        for region in metrics_by_region_then_field_name.keys()
        for variable in metrics_by_region_then_field_name[region].keys()
    }

    # Make sure there is only one len among all of loc_var_seq.values(). Make a DatetimeIndex
    # with that many dates.
    sequence_lengths = more_itertools.one(set(len(seq) for seq in loc_var_seq.values()))
    dates = pd.date_range(start_date, periods=sequence_lengths, freq="D", name=CommonFields.DATE)

    index = pd.MultiIndex.from_tuples(
        loc_var_seq.keys(), names=[CommonFields.LOCATION_ID, PdFields.VARIABLE],
    )

    df = pd.DataFrame(list(loc_var_seq.values()), index=index, columns=dates)
    df = df.fillna(np.nan).apply(pd.to_numeric)

    dataset = timeseries.MultiRegionDataset.from_timeseries_wide_dates_df(df)

    if timeseries_columns:
        new_timeseries = _add_missing_columns(dataset.timeseries, timeseries_columns)
        dataset = dataclasses.replace(dataset, timeseries=new_timeseries)

    tags_to_concat = []
    for (loc_id, var), ts_literal in loc_var_seq.items():
        if not isinstance(ts_literal, TimeseriesLiteral):
            continue

        records = list(ts_literal.annotation)
        if ts_literal.provenance:
            records.append((timeseries.TagType.PROVENANCE, pd.NaT, ts_literal.provenance))
        df = pd.DataFrame.from_records(
            records, columns=[TagField.TYPE, TagField.DATE, TagField.CONTENT]
        )
        df[TagField.DATE] = pd.to_datetime(df[TagField.DATE])
        df[TagField.LOCATION_ID] = loc_id
        df[TagField.VARIABLE] = var
        tags_to_concat.append(df)

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
