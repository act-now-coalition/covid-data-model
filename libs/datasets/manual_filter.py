import dataclasses

import structlog
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields
import pandas as pd

from libs.datasets import AggregationLevel
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.pipeline import Region, RegionMask


_logger = structlog.getLogger()


# TODO(tom): Make some kind of hierarchy of class to form a schema that can be populated by a JSON.
CONFIG = {
    "filters": [
        {
            "regions_included": [RegionMask(AggregationLevel.COUNTY, states=["OK"])],
            "regions_excluded": [Region.from_fips("40109"), Region.from_fips("40143")],
            "observations_to_drop": {
                "start_date": "2021-03-15",
                "fields": [CommonFields.CASES, CommonFields.DEATHS],
                "internal_note": "https://trello.com/c/HdAKfp49/1139",
                "public_note": "Something broke with the OK county data.",
            },
        },
    ]
}


def drop_observations(
    dataset: timeseries.MultiRegionDataset, config
) -> timeseries.MultiRegionDataset:
    """Drops observations according to `config` from every region in dataset."""
    drop_start_date = pd.to_datetime(config["start_date"])

    ts_in = dataset.timeseries_bucketed_wide_dates

    mask_selected_fields = ts_in.index.get_level_values(PdFields.VARIABLE).isin(config["fields"])
    ts_selected_fields = ts_in.loc[mask_selected_fields]
    ts_not_selected_fields = ts_in.loc[~mask_selected_fields]

    obsv_selected = ts_selected_fields.loc[:, ts_selected_fields.columns >= drop_start_date]
    mask_has_real_value_to_drop = obsv_selected.notna().any(1)
    ts_to_filter = ts_selected_fields.loc[mask_has_real_value_to_drop]
    ts_no_real_values_to_drop = ts_selected_fields.loc[~mask_has_real_value_to_drop]

    ts_filtered = ts_to_filter.loc[:, ts_to_filter.columns < drop_start_date]

    new_tags = taglib.TagCollection()
    assert ts_to_filter.index.names == [
        CommonFields.LOCATION_ID,
        PdFields.VARIABLE,
        PdFields.DEMOGRAPHIC_BUCKET,
    ]
    for location_id, variable, bucket in ts_to_filter.index:
        new_tags.add(
            taglib.KnownIssue(date=drop_start_date, disclaimer=config["public_note"]),
            location_id=location_id,
            variable=variable,
            bucket=bucket,
        )

    ts_new = (
        pd.concat([ts_not_selected_fields, ts_no_real_values_to_drop, ts_filtered])
        .stack()
        .unstack(PdFields.VARIABLE)
        .sort_index()
    )
    return dataclasses.replace(dataset, timeseries_bucketed=ts_new).append_tag_df(
        new_tags.as_dataframe()
    )


def run(dataset: timeseries.MultiRegionDataset, config=CONFIG) -> timeseries.MultiRegionDataset:
    for filter_ in config["filters"]:
        filtered_dataset = dataset.get_regions_subset(filter_["regions_included"])
        if filter_.get("regions_excluded"):
            filtered_dataset = filtered_dataset.remove_regions(filter_["regions_excluded"])
        if filtered_dataset.location_ids.empty:
            # TODO(tom): Find a cleaner way to refer to a filter in logs.
            _logger.info("No locations matched", regions=str(filter_["regions_included"]))
            continue
        passed_dataset = dataset.remove_locations(filtered_dataset.location_ids)
        observations_to_drop = filter_.get("observations_to_drop")
        filtered_dataset = drop_observations(filtered_dataset, observations_to_drop)

        dataset = filtered_dataset.append_regions(passed_dataset)

    return dataset
