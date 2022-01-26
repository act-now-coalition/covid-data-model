import dataclasses
from typing import Collection

import structlog
from datapublic.common_fields import CommonFields
from datapublic.common_fields import PdFields

from libs.datasets import timeseries
import pandas as pd


_log = structlog.get_logger()


DROPPING_TIMESERIES_WITH_ONLY_ZEROS = "Dropping timeseries with only zeros"


def drop_all_zero_timeseries(
    ds_in: timeseries.MultiRegionDataset, fields: Collection[CommonFields]
) -> timeseries.MultiRegionDataset:
    """Returns a dataset with `fields` timeseries dropped if they contain only NA and 0.

    When first built this is dropping a timeseries in Loving County, TX which has so few people
    that the all-zero timeseries is likely accurate. It may be worth only applying this to
    locations with a population over some threshold. Or perhaps an automatic filter isn't worth
    the trouble after all :-(
    """
    ts_wide = ds_in.timeseries_bucketed_wide_dates

    # Separate into timeseries in `fields` and all others.
    variable_mask = ts_wide.index.get_level_values(PdFields.VARIABLE).isin(fields)
    ts_wide_other_variables = ts_wide.loc[~variable_mask]
    ts_wide_variables = ts_wide.loc[variable_mask]

    # Keep rows/timeseries that have at least one value that is not 0 or NA
    to_keep_mask = ts_wide_variables.replace(pd.NA, 0).any(axis=1)
    to_drop = ts_wide_variables.loc[~to_keep_mask].index
    if not to_drop.empty:
        # Maybe add filtering to not log about the known bad data in OH counties and Loving
        # County Texas using a RegionMask(level=County, state=OH) and some kind of RegionMask
        # representing counties with a small population.
        _log.info(DROPPING_TIMESERIES_WITH_ONLY_ZEROS, dropped=to_drop)
    ts_wide_kept = ts_wide_variables.loc[to_keep_mask]

    ts_wide_out = pd.concat([ts_wide_kept, ts_wide_other_variables])

    # Make a new dataset without the dropped timeseries. This does not drop the tags of the
    # dropped timeseries but keeping the provenance tags doesn't seem to be a problem. Maybe it'd
    # be cleaner to add a method 'MultiRegionDataset.drop_timeseries' similar to 'remove_regions' or
    # move this into 'MultiRegionDataset' similar to 'drop_stale_timeseries'.
    return dataclasses.replace(
        ds_in, timeseries_bucketed=ts_wide_out.stack().unstack(PdFields.VARIABLE).sort_index(),
    )
