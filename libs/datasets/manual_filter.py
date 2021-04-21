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


# TODO(tom): Make some kind of hierarchy of class to form a schema that can be populated by a JSON.
CONFIG = {
    "filters": [
        {
            "regions_included": [RegionMask(AggregationLevel.COUNTY, states=["OK"])],
            "regions_excluded": [Region.from_fips("40109"), Region.from_fips("40143")],
            "observations_to_drop": {
                "start_date": "2021-03-15",
                "drop_fields": [CommonFields.CASES, CommonFields.DEATHS],
                "internal_note": "https://trello.com/c/HdAKfp49/1139",
                "public_note": "Something broke with the OK county data.",
            },
        },
    ]
}


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


def _partition_by_fields(
    dataset: timeseries.MultiRegionDataset, fields: List[CommonFields]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns a tuple of the timeseries in fields and all others, in wide dates DataFrames."""
    ts_in = dataset.timeseries_bucketed_wide_dates
    mask_selected_fields = ts_in.index.get_level_values(PdFields.VARIABLE).isin(fields)
    ts_selected_fields = ts_in.loc[mask_selected_fields]
    ts_not_selected_fields = ts_in.loc[~mask_selected_fields]
    return ts_selected_fields, ts_not_selected_fields


def _filter_by_date(
    ts_in: pd.DataFrame, *, drop_start_date: datetime.date
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_date = pd.to_datetime(drop_start_date)
    obsv_selected = ts_in.loc[:, ts_in.columns >= start_date]
    mask_has_real_value_to_drop = obsv_selected.notna().any(1)
    ts_to_filter = ts_in.loc[mask_has_real_value_to_drop]
    ts_no_real_values_to_drop = ts_in.loc[~mask_has_real_value_to_drop]
    ts_filtered = ts_to_filter.loc[:, ts_to_filter.columns < start_date]
    return ts_no_real_values_to_drop, ts_filtered


def drop_observations(
    dataset: timeseries.MultiRegionDataset,
    fields: List[CommonFields],
    drop_start_date: datetime.date,
    public_note: str,
) -> timeseries.MultiRegionDataset:
    """Drops observations according to parameters from every region in `dataset`."""
    ts_selected_fields, ts_not_selected_fields = _partition_by_fields(dataset, fields)

    ts_no_real_values_to_drop, ts_filtered = _filter_by_date(
        ts_selected_fields, drop_start_date=drop_start_date
    )

    tag = taglib.KnownIssue(date=drop_start_date, disclaimer=public_note)

    new_tags = taglib.TagCollection()
    new_tags.add_by_index(tag, index=ts_filtered.index)

    return dataset.replace_timeseries_bucketed(
        [ts_not_selected_fields, ts_no_real_values_to_drop, ts_filtered]
    ).append_tag_df(new_tags.as_dataframe())


def run(dataset: timeseries.MultiRegionDataset, config=CONFIG) -> timeseries.MultiRegionDataset:
    config = Config.parse_obj(config)
    for filter_ in config.filters:
        filtered_dataset = dataset.get_regions_subset(filter_.regions_included)
        if filter_.regions_excluded:
            filtered_dataset = filtered_dataset.remove_regions(filter_.regions_excluded)
        if filtered_dataset.location_ids.empty:
            # TODO(tom): Find a cleaner way to refer to a filter in logs.
            _logger.info("No locations matched", regions=str(filter_.regions_included))
            continue
        passed_dataset = dataset.remove_locations(filtered_dataset.location_ids)
        filtered_dataset = drop_observations(
            filtered_dataset,
            filter_.observations_to_drop.drop_field_list,
            filter_.observations_to_drop.start_date,
            filter_.observations_to_drop.public_note,
        )

        dataset = filtered_dataset.append_regions(passed_dataset)

    return dataset
