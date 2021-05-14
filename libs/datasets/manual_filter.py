import datetime
from typing import Optional
from typing import Tuple
import re
from typing import List
from typing import Mapping


import pydantic
import structlog
from covidactnow.datapublic import common_fields
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldGroup
from covidactnow.datapublic.common_fields import PdFields
import pandas as pd

from libs.datasets import AggregationLevel
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.pipeline import Region, RegionMask
from libs.pipeline import RegionMaskOrRegion

_logger = structlog.getLogger()


# Tag types expected to be produced by this module and only this module.
_EXPECTED_TYPES = [taglib.TagType.KNOWN_ISSUE, taglib.TagType.KNOWN_ISSUE_NO_DATE]


# The manual filter config is defined with pydantic instead of standard dataclasses to get support
# for parsing from json, automatic conversion of fields to the expected type and avoiding the
# dreaded "TypeError: non-default argument ...". pydantic reserves the field name 'fields'.
class Filter(pydantic.BaseModel):
    regions_included: List[RegionMaskOrRegion]
    regions_excluded: Optional[List[RegionMaskOrRegion]] = None
    fields_included: List[CommonFields]
    start_date: Optional[datetime.date] = None
    drop_observations: bool
    internal_note: str
    # public_note may be any empty string
    public_note: str

    @property
    def tag(self) -> taglib.TagInTimeseries:
        if self.start_date:
            return taglib.KnownIssue(date=self.start_date, disclaimer=self.public_note)
        else:
            return taglib.KnownIssueNoDate(public_note=self.public_note)

    # From https://github.com/samuelcolvin/pydantic/issues/568 ... pylint: disable=no-self-argument
    @pydantic.root_validator()
    def check(cls, values):
        if not values["drop_observations"]:
            if values["public_note"] == "":
                raise ValueError(
                    "Filter doesn't drop observations or add a public_note. What does it do?"
                )
            if values["start_date"]:
                raise ValueError(
                    "Filter including start_date without dropping observations "
                    "doesn't make sense."
                )
        return values


class Config(pydantic.BaseModel):
    filters: List[Filter]


CONFIG = Config(filters=[])


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
    dataset: timeseries.MultiRegionDataset, filter_: Filter
) -> timeseries.MultiRegionDataset:
    """Drops observations according to `filter_` from every region in `dataset`."""
    assert filter_.drop_observations

    # wide-dates DataFrames that will be concat-ed to produce the result MultiRegionDataset.
    ts_results = []
    ts_selected_fields, ts_not_selected_fields = _partition_by_fields(
        dataset.timeseries_bucketed_wide_dates, filter_.fields_included
    )
    ts_results.append(ts_not_selected_fields)

    if filter_.start_date:
        ts_filtered, ts_no_real_values_to_drop = _filter_by_date(
            ts_selected_fields, drop_start_date=filter_.start_date
        )
        return dataset.replace_timeseries_wide_dates(
            [ts_not_selected_fields, ts_filtered, ts_no_real_values_to_drop]
        ).add_tag_to_subset(filter_.tag, ts_filtered.index)
    else:
        # When start_date is None all of ts_selected_fields is dropped; only the not selected
        # fields are kept.
        selected_index = ts_selected_fields.index
        return (
            dataset.replace_timeseries_wide_dates([ts_not_selected_fields])
            .remove_tags_from_subset(selected_index)
            .add_tag_to_subset(filter_.tag, selected_index)
        )


def run(
    dataset: timeseries.MultiRegionDataset, config: Config = CONFIG
) -> timeseries.MultiRegionDataset:
    for filter_ in config.filters:
        assert filter_.tag.TAG_TYPE in _EXPECTED_TYPES
        filtered_dataset, passed_dataset = dataset.partition_by_region(
            filter_.regions_included, exclude=filter_.regions_excluded
        )
        if filtered_dataset.location_ids.empty:
            # TODO(tom): Find a cleaner way to refer to a filter in logs.
            _logger.info("No locations matched", regions=str(filter_.regions_included))
            continue
        if filter_.drop_observations:
            filtered_dataset = drop_observations(filtered_dataset, filter_)
        else:
            ts_selected_fields, _ = _partition_by_fields(
                filtered_dataset.timeseries_bucketed_wide_dates, filter_.fields_included
            )
            filtered_dataset = filtered_dataset.add_tag_to_subset(
                filter_.tag, ts_selected_fields.index
            )

        dataset = filtered_dataset.append_regions(passed_dataset)

    return dataset


def touched_subset(
    ds_in: timeseries.MultiRegionDataset, ds_out: timeseries.MultiRegionDataset
) -> timeseries.MultiRegionDataset:
    """Given the input and output of `run`, returns the time series from ds_in that have tags in
    ds_out added by `run` combined with tags of these time series copied from ds_out."""
    # Expected tags do not appear in the input
    assert ds_in.tag.index.unique(taglib.TagField.TYPE).intersection(_EXPECTED_TYPES).empty
    tag_mask = ds_out.tag.index.get_level_values(taglib.TagField.TYPE).isin(_EXPECTED_TYPES)
    touch_index = ds_out.tag.index[tag_mask].droplevel(taglib.TagField.TYPE).unique()
    wide_dates_in_df = ds_in.timeseries_bucketed_wide_dates
    # Get all ds_in.tag associated with any time series in touch_index
    tag_in = ds_in.tag.loc[ds_in.tag.index.droplevel(taglib.TagField.TYPE).isin(touch_index)]
    # Get only the ds_out.tag where the type is _EXPECTED_TYPES. These are the tags added by
    # `run` and not in ds_in.tag.
    tag_out = ds_out.tag.loc[tag_mask]
    assert wide_dates_in_df.index.names == touch_index.names
    assert touch_index.isin(wide_dates_in_df.index).all()
    return (
        timeseries.MultiRegionDataset.from_timeseries_wide_dates_df(
            wide_dates_in_df.reindex(touch_index), bucketed=True
        )
        .append_tag_df(tag_in.reset_index())
        .append_tag_df(tag_out.reset_index())
    )


# Possible values from data/region-overrides.json
_METRIC_TO_FIELDS = {
    "metrics.caseDensity": [CommonFields.CASES, CommonFields.NEW_CASES],
    # infectionRate is ignored without error. It isn't used in the current region-overrides.json and
    # doesn't exist as a field in the pipeline where the manual_filter is applied. Removing cases
    # will likely cause Rt to no longer be produced for a region.
    "metrics.infectionRate": [],
    "metrics.testPositivityRatio": common_fields.FIELD_GROUP_TO_LIST_FIELDS[FieldGroup.TESTS],
    "metrics.icuCapacityRatio": common_fields.FIELD_GROUP_TO_LIST_FIELDS[
        FieldGroup.HEALTHCARE_CAPACITY
    ],
    "metrics.vaccinationsInitiatedRatio": common_fields.FIELD_GROUP_TO_LIST_FIELDS[
        FieldGroup.VACCINES
    ],
}


def _transform_one_override(
    override: Mapping, cbsa_to_counties_map: Mapping[Region, List[Region]]
) -> Filter:
    region_str = override["region"]
    if re.fullmatch(r"[A-Z][A-Z]", region_str):
        region = Region.from_state(region_str)
    elif re.fullmatch(r"\d{5}", region_str):
        region = Region.from_fips(region_str)
    else:
        raise ValueError(f"Invalid region: {region_str}")

    include_str = override["include"]
    if include_str == "region":
        regions_included = [region]
    elif include_str == "region-and-subregions":
        if region.is_state():
            regions_included = [RegionMask(states=[region.state])]
        elif region.level == AggregationLevel.CBSA:
            regions_included = [region] + cbsa_to_counties_map[region]
        else:
            raise ValueError("region-and-subregions only valid for a state and CBSA")
    elif include_str == "subregions":
        if not region.is_state():
            raise ValueError("subregions only valid for a state")
        regions_included = [RegionMask(AggregationLevel.COUNTY, states=[region.state])]
    else:
        raise ValueError(f"Invalid include: {include_str}")

    return Filter(
        regions_included=regions_included,
        fields_included=_METRIC_TO_FIELDS[override["metric"]],
        internal_note=override["context"],
        public_note=override.get("disclaimer", ""),
        drop_observations=bool(override["blocked"]),
    )


def transform_region_overrides(
    region_overrides: Mapping, cbsa_to_counties_map: Mapping[Region, List[Region]]
) -> Config:
    filter_configs: List[Filter] = []
    for override in region_overrides["overrides"]:
        try:
            filter_configs.append(_transform_one_override(override, cbsa_to_counties_map))
        except:
            raise ValueError(f"Problem with {override}")

    return Config(filters=filter_configs)
