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
from libs.pipeline import Region


# This is a totally bogus fips/region/location that we've been using as a default in some test
# cases. It is factored out here in an attempt to reduce how much it is hard-coded into our source.
DEFAULT_FIPS = "97222"
DEFAULT_REGION = Region.from_fips(DEFAULT_FIPS)


class TimeseriesLiteral(UserList):
    """Represents a timeseries literal, a sequence of floats and provenance string."""

    def __init__(
        self, ts_list, *, provenance: str = "",
    ):
        super().__init__(ts_list)
        self.provenance = provenance


def build_dataset(
    metrics_by_region_then_field_name: Mapping[
        Region, Mapping[FieldName, Union[Sequence[float], TimeseriesLiteral]]
    ],
    *,
    start_date="2020-04-01",
) -> timeseries.MultiRegionDataset:
    """Returns a dataset for multiple regions and metrics.
    Args:
        metrics_by_region_then_field_name: Each sequence of values and TimeseriesLiteral must have
            at least one real value and identical length. The oldest date is the 0th element.
        start_date: The oldest date of each timeseries.
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

    loc_var_provenance = {
        key: ts_lit.provenance
        for key, ts_lit in loc_var_seq.items()
        if isinstance(ts_lit, TimeseriesLiteral)
    }
    if loc_var_provenance:
        provenance_index = pd.MultiIndex.from_tuples(
            loc_var_provenance.keys(), names=[CommonFields.LOCATION_ID, PdFields.VARIABLE]
        )
        provenance_series = pd.Series(
            list(loc_var_provenance.values()),
            dtype="str",
            index=provenance_index,
            name=PdFields.PROVENANCE,
        )
        dataset = dataset.add_provenance_series(provenance_series)

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
    one_region = build_dataset({region: metrics}, start_date=start_date).get_one_region(region)
    if timeseries_columns:
        new_columns = [col for col in timeseries_columns if col not in one_region.data.columns]
        new_data = one_region.data.reindex(columns=[*one_region.data.columns, *new_columns])
        one_region = dataclasses.replace(one_region, data=new_data)
    if latest_override:
        new_latest = {**one_region.latest, **latest_override}
        one_region = dataclasses.replace(one_region, latest=new_latest)
    return one_region
