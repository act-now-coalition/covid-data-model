import dataclasses

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldGroup
from covidactnow.datapublic.common_fields import PdFields
import pandas as pd

from libs.datasets import AggregationLevel
from libs.datasets import timeseries
from libs.pipeline import Region, RegionMask


CONFIG = {
    "filters": [
        {
            "regions_included": [Region.from_fips("45001"), Region.from_fips("45003")],
            "drop_observations": [
                {
                    "start_date": "2021-02-14",
                    "fields": [CommonFields.CASES],
                    "internal_note": "something happened",
                    "public_note": "everything is great",
                }
            ],
        }
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
        dataset_pass = dataset.remove_regions(filter_["regions_included"])
        for drop_config in filter_["drop_observations"]:
            dataset_filter = drop_observations(dataset_filter, drop_config)

        dataset = dataset_filter.append_regions(dataset_pass)

    return dataset
