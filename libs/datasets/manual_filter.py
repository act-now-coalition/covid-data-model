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
    end_date: Optional[datetime.date] = None
    drop_observations: bool
    internal_note: str
    # public_note may be any empty string
    public_note: str

    @property
    def tag(self) -> taglib.TagInTimeseries:
        if self.start_date:
            return taglib.KnownIssue(date=self.start_date, public_note=self.public_note)
        elif self.end_date:
            return taglib.KnownIssue(date=self.end_date, public_note=self.public_note)
        else:
            return taglib.KnownIssueNoDate(public_note=self.public_note)

    # From https://github.com/samuelcolvin/pydantic/issues/568 ... pylint: disable=no-self-argument
    @pydantic.root_validator()
    def check(cls, values):
        start_date = values["start_date"]
        end_date = values["end_date"]
        if not values["drop_observations"]:
            if values["public_note"] == "":
                raise ValueError(
                    "Filter doesn't drop observations or add a public_note. What does it do?"
                )
            if start_date:
                raise ValueError(
                    "Filter including start_date without dropping observations doesn't make sense."
                )
            if end_date:
                raise ValueError(
                    "Filter including end_date without dropping observations doesn't make sense."
                )
        if start_date and end_date and end_date < start_date:
            raise ValueError(f"Filter end_date ({end_date}) is before start_date ({start_date})")

        return values


class Config(pydantic.BaseModel):
    filters: List[Filter]


def _partition_by_fields(
    ts_in: pd.DataFrame, fields: List[CommonFields]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions a DataFrame with timeseries wide dates into two: fields and others."""
    timeseries.check_timeseries_wide_dates_structure(ts_in)

    # NOTE(sean): By default we only block non-demographic data. The demographic
    # data does not reach the website, but is used by partners so we want this to flow through,
    # regardless of block status.
    mask_selected_fields = ts_in.index.get_level_values(PdFields.VARIABLE).isin(
        fields
    ) & ts_in.index.get_level_values(PdFields.DEMOGRAPHIC_BUCKET).isin(["all"])
    ts_selected_fields = ts_in.loc[mask_selected_fields]
    ts_not_selected_fields = ts_in.loc[~mask_selected_fields]
    return ts_selected_fields, ts_not_selected_fields


def _filter_by_date(
    ts_in: pd.DataFrame,
    *,
    drop_start_date: Optional[datetime.date],
    drop_end_date: Optional[datetime.date],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Removes observations from specified dates, returning the modified timeseries and
    unmodified timeseries in separate DataFrame objects."""
    timeseries.check_timeseries_wide_dates_structure(ts_in)

    drop_masks = []  # Masks for ts_in.columns. A date is dropped when True in all masks.
    if drop_start_date:
        drop_masks.append(pd.to_datetime(drop_start_date) <= ts_in.columns)
    if drop_end_date:
        drop_masks.append(ts_in.columns <= pd.to_datetime(drop_end_date))
    if len(drop_masks) == 1:
        columns_to_drop_mask = drop_masks[0]  # Only one of start and end date was set
    else:
        assert len(drop_masks) == 2  # Both start and end date were set
        columns_to_drop_mask = drop_masks[0] & drop_masks[1]

    observations_dropped = ts_in.loc[:, columns_to_drop_mask]
    ts_has_real_value_to_drop_mask = observations_dropped.notna().any(1)
    ts_to_filter = ts_in.loc[ts_has_real_value_to_drop_mask]
    ts_no_real_values_to_drop = ts_in.loc[~ts_has_real_value_to_drop_mask]
    ts_filtered = ts_to_filter.loc[:, ~columns_to_drop_mask]
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

    if filter_.start_date or filter_.end_date:
        ts_filtered, ts_no_real_values_to_drop = _filter_by_date(
            ts_selected_fields, drop_start_date=filter_.start_date, drop_end_date=filter_.end_date
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


def run(dataset: timeseries.MultiRegionDataset, config: Config) -> timeseries.MultiRegionDataset:
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
    ds_out added by `run` and dropped observations, combined with tags of these time series
    copied from ds_out."""
    # Check that tags added by `run` do not appear in the input of `run`.
    assert ds_in.tag.index.unique(taglib.TagField.TYPE).intersection(_EXPECTED_TYPES).empty
    tag_mask = ds_out.tag.index.get_level_values(taglib.TagField.TYPE).isin(_EXPECTED_TYPES)
    # ts_has_tag_index identifies the subset of time series rows that have an _EXPECTED_TYPES tag.
    ts_has_tag_index = ds_out.tag.index[tag_mask].droplevel(taglib.TagField.TYPE).unique()
    wide_dates_in_df = ds_in.timeseries_bucketed_wide_dates
    assert wide_dates_in_df.index.names == ts_has_tag_index.names
    assert ts_has_tag_index.isin(wide_dates_in_df.index).all()
    # For time series identified by ts_has_tag_index find the number of observations per time
    # series in ds_in and ds_out.
    in_observations_by_ts = wide_dates_in_df.reindex(ts_has_tag_index).notna().sum(axis="columns")
    out_observations_by_ts = (
        ds_out.timeseries_bucketed_wide_dates.reindex(ts_has_tag_index).notna().sum(axis="columns")
    )
    # Make an index which identifies time series that had a decrease in number of observations
    # from ds_in to ds_out.
    ts_lost_observation_mask = out_observations_by_ts < in_observations_by_ts
    ts_lost_observation_index = in_observations_by_ts.loc[ts_lost_observation_mask].index

    def tag_xs(tag: pd.Series, ts_index: pd.MultiIndex) -> pd.Series:
        """Return the cross-section of `tag` that has index labels in `ts_index`."""
        tag_ts_index = tag.index.droplevel([taglib.TagField.TYPE])
        assert tag_ts_index.names == ts_index.names
        return tag.loc[tag_ts_index.isin(ts_index)]

    # Get all ds_in.tag associated with any time series in ts_lost_observation_index.
    tag_in = tag_xs(ds_in.tag, ts_lost_observation_index)
    # Get only the ds_out.tag where the type is _EXPECTED_TYPES and observations were lost. This
    # avoids duplicating tags in tag_in.
    tag_out = tag_xs(ds_out.tag.loc[tag_mask], ts_lost_observation_index)
    return (
        timeseries.MultiRegionDataset.from_timeseries_wide_dates_df(
            wide_dates_in_df.reindex(ts_lost_observation_index), bucketed=True
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
    include_str = override["include"]

    if not isinstance(region_str, str):
        raise ValueError(f"Invalid 'region': {region_str}")
    if not isinstance(include_str, str):
        raise ValueError(f"Invalid 'include': {include_str}")

    regions_included = _transform_region_str(region_str, include_str, cbsa_to_counties_map)

    # The CMS stores empty strings when no date is specified, hence "or None"
    start_date = override.get("start_date", None) or None
    end_date = override.get("end_date", None) or None

    return Filter(
        regions_included=regions_included,
        fields_included=_METRIC_TO_FIELDS[override["metric"]],
        internal_note=override["context"],
        public_note=override.get("disclaimer", ""),
        drop_observations=bool(override["blocked"]),
        start_date=start_date,
        end_date=end_date,
    )


def _transform_region_str(
    raw_region_str: str, include_str: str, cbsa_to_counties_map: Mapping[Region, List[Region]]
) -> List[RegionMaskOrRegion]:

    # We allow multiple regions to be specified, separated by commas.
    region_strs = [single_region_str.strip() for single_region_str in raw_region_str.split(",")]

    regions_included = []
    for region_str in region_strs:
        if re.fullmatch(r"[A-Z][A-Z]", region_str):
            region = Region.from_state(region_str)
        elif re.fullmatch(r"\d{5}", region_str):
            region = Region.from_fips(region_str)
        else:
            raise ValueError(f"Invalid region: {region_str}")

        if include_str == "region":
            regions_included.append(region)
        elif include_str == "region-and-subregions":
            if region.is_state():
                regions_included.append(RegionMask(states=[region.state]))
            elif region.level == AggregationLevel.CBSA:
                regions_included.extend([region] + cbsa_to_counties_map[region])
            else:
                raise ValueError("region-and-subregions only valid for a state and CBSA")
        elif include_str == "subregions":
            if not region.is_state():
                raise ValueError("subregions only valid for a state")
            regions_included.append(RegionMask(AggregationLevel.COUNTY, states=[region.state]))
        else:
            raise ValueError(f"Invalid include: {include_str}")

    return regions_included


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
