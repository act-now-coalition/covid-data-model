import datetime
from typing import List
from typing import Optional
from typing import Tuple

import pydantic
import structlog
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields
import pandas as pd

from libs.datasets import AggregationLevel
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.pipeline import Region, RegionMask
from libs.pipeline import RegionMaskOrRegion

_logger = structlog.getLogger()


# The manual filter config is defined with pydantic instead of standard dataclasses to get support
# for parsing from json, automatic conversion of fields to the expected type and fields that default
# to None. pydantic reserves the field name 'fields'.


class ObservationsToDrop(pydantic.BaseModel):
    start_date: datetime.date
    drop_fields: Optional[List[CommonFields]] = None
    drop_field_group: Optional[common_fields.FieldGroup] = None
    internal_note: str
    public_note: str

    @property
    def drop_field_list(self) -> List[CommonFields]:
        if self.drop_fields is not None:
            assert self.drop_field_group is None
            return self.drop_fields
        else:
            # self.drop_fields is None
            assert self.drop_field_group is not None
            return common_fields.FIELD_GROUP_TO_LIST_FIELDS[self.drop_field_group]


class Filter(pydantic.BaseModel):
    regions_included: List[RegionMaskOrRegion]
    regions_excluded: Optional[List[RegionMaskOrRegion]] = None
    observations_to_drop: ObservationsToDrop


class Config(pydantic.BaseModel):
    filters: List[Filter]


CONFIG = Config(
    filters=[
        Filter(
            regions_included=[RegionMask(AggregationLevel.COUNTY, states=["OK"])],
            regions_excluded=[Region.from_fips("40109"), Region.from_fips("40143")],
            observations_to_drop=ObservationsToDrop(
                start_date="2021-03-15",
                drop_fields=[CommonFields.CASES, CommonFields.DEATHS],
                internal_note="https://trello.com/c/HdAKfp49/1139",
                public_note="Something broke with the OK county data.",
            ),
        )
    ]
)


def _partition_by_fields(
    ts_in: pd.DataFrame, fields: List[CommonFields]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions a DataFrame with timeseries wide dates into two: fields and others."""
    timeseries.check_timeseries_wide_dates_structure(ts_in)
    mask_selected_fields = ts_in.index.get_level_values(PdFields.VARIABLE).isin(fields)
    ts_selected_fields = ts_in.loc[mask_selected_fields]
    ts_not_selected_fields = ts_in.loc[~mask_selected_fields]
    return ts_selected_fields, ts_not_selected_fields


def _filter_by_date(
    ts_in: pd.DataFrame, *, drop_start_date: datetime.date
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Removes observations from specified dates, returning the modified timeseries and
    unmodified timeseries in separate DataFrame objects."""
    timeseries.check_timeseries_wide_dates_structure(ts_in)
    start_date = pd.to_datetime(drop_start_date)
    obsv_selected = ts_in.loc[:, ts_in.columns >= start_date]
    mask_has_real_value_to_drop = obsv_selected.notna().any(1)
    ts_to_filter = ts_in.loc[mask_has_real_value_to_drop]
    ts_no_real_values_to_drop = ts_in.loc[~mask_has_real_value_to_drop]
    ts_filtered = ts_to_filter.loc[:, ts_to_filter.columns < start_date]
    return ts_filtered, ts_no_real_values_to_drop


def drop_observations(
    dataset: timeseries.MultiRegionDataset, config: ObservationsToDrop
) -> timeseries.MultiRegionDataset:
    """Drops observations according to `config` from every region in `dataset`."""
    ts_selected_fields, ts_not_selected_fields = _partition_by_fields(
        dataset.timeseries_bucketed_wide_dates, config.drop_field_list
    )

    ts_filtered, ts_no_real_values_to_drop = _filter_by_date(
        ts_selected_fields, drop_start_date=config.start_date
    )

    tag = taglib.KnownIssue(date=config.start_date, disclaimer=config.public_note)

    new_tags = taglib.TagCollection()
    new_tags.add_by_index(tag, index=ts_filtered.index)

    return dataset.replace_timeseries_wide_dates(
        [ts_not_selected_fields, ts_no_real_values_to_drop, ts_filtered]
    ).append_tag_df(new_tags.as_dataframe())


def run(
    dataset: timeseries.MultiRegionDataset, config: Config = CONFIG
) -> timeseries.MultiRegionDataset:
    for filter_ in config.filters:
        filtered_dataset, passed_dataset = dataset.partition_by_region(
            filter_.regions_included, exclude=filter_.regions_excluded
        )
        if filtered_dataset.location_ids.empty:
            # TODO(tom): Find a cleaner way to refer to a filter in logs.
            _logger.info("No locations matched", regions=str(filter_.regions_included))
            continue
        filtered_dataset = drop_observations(filtered_dataset, filter_.observations_to_drop)

        dataset = filtered_dataset.append_regions(passed_dataset)

    return dataset
