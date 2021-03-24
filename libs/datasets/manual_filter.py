import dataclasses

import structlog
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields
import pandas as pd

from libs.datasets import AggregationLevel
from libs.datasets import timeseries
from libs.pipeline import Region, RegionMask


_logger = structlog.getLogger()


CONFIG = {
    "filters": [
        {
            "regions_included": [
                Region.from_fips("49009"),
                Region.from_fips("49013"),
                Region.from_fips("49047"),
            ],
            "drop_observations": [
                {
                    "start_date": "2021-02-12",
                    "fields": [CommonFields.CASES, CommonFields.DEATHS],
                    "internal_note": "https://trello.com/c/aj7ep7S7/1130",
                    "public_note": "The TriCounty Health Department is focusing on vaccinations "
                    "and we have not found a new source of case counts.",
                }
            ],
        },
        {
            "regions_included": [RegionMask(AggregationLevel.COUNTY, states=["OK"])],
            "regions_excluded": [Region.from_fips("40109"), Region.from_fips("40143")],
            "drop_observations": [
                {
                    "start_date": "2021-03-15",
                    "fields": [CommonFields.CASES, CommonFields.DEATHS],
                    "internal_note": "https://trello.com/c/HdAKfp49/1139",
                    "public_note": "Something broke with the OK county data.",
                }
            ],
        },
    ]
}


def drop_observations(
    dataset: timeseries.MultiRegionDataset, config
) -> timeseries.MultiRegionDataset:
    drop_start_date = pd.to_datetime(config["start_date"])

    timeseries_wide_dates = dataset.timeseries_bucketed_wide_dates
    ts_mask = timeseries_wide_dates.index.get_level_values(PdFields.VARIABLE).isin(config["fields"])
    ts_filter = timeseries_wide_dates[ts_mask]
    ts_pass = timeseries_wide_dates[~ts_mask]

    ts_filter = ts_filter.loc[:, ts_filter.columns < drop_start_date]

    ts_new = pd.concat([ts_filter, ts_pass]).stack().unstack(PdFields.VARIABLE).sort_index()
    return dataclasses.replace(dataset, timeseries_bucketed=ts_new)


def run(dataset: timeseries.MultiRegionDataset, config=CONFIG) -> timeseries.MultiRegionDataset:
    for filter_ in config["filters"]:
        dataset_filter = dataset.get_regions_subset(filter_["regions_included"])
        if filter_.get("regions_excluded"):
            dataset_filter = dataset_filter.remove_regions(filter_["regions_excluded"])
        if dataset_filter.location_ids.empty:
            _logger.info("No locations matched", regions=str(filter_["regions_included"]))
            continue
        dataset_pass = dataset.remove_locations(dataset_filter.location_ids)
        for drop_config in filter_["drop_observations"]:
            dataset_filter = drop_observations(dataset_filter, drop_config)

        dataset = dataset_filter.append_regions(dataset_pass)

    return dataset
