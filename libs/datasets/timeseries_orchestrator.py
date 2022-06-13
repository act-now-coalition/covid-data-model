from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Mapping
from libs.datasets import dataset_utils
from libs.datasets import timeseries
from libs.datasets import combined_datasets
from datapublic.common_fields import FieldName
import pandas as pd

from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.combined_datasets import (
    ALL_TIMESERIES_FEATURE_DEFINITION,
    ALL_FIELDS_FEATURE_DEFINITION,
)
from libs.parallel_utils import parallel_map
from libs.us_state_abbrev import US_STATE_ABBREV

from libs.datasets import manual_filter
from libs.datasets import combined_dataset_utils, custom_patches, weekly_hospitalizations

import datetime
from typing import List
from typing import Mapping
from typing import Optional
import structlog

from datapublic.common_fields import CommonFields, PdFields
from datapublic.common_fields import FieldName

from libs.datasets import custom_patches, weekly_hospitalizations
from libs.datasets import nytimes_anomalies
from libs.datasets import custom_aggregations
from libs.datasets import statistical_areas
from libs.datasets.combined_datasets import (
    ALL_TIMESERIES_FEATURE_DEFINITION,
    ALL_FIELDS_FEATURE_DEFINITION,
)
from libs.datasets import timeseries
from libs.datasets import outlier_detection
from libs.datasets import dataset_utils
from libs.datasets import combined_datasets
from libs.datasets import new_cases_and_deaths
from libs.datasets import vaccine_backfills
from libs.datasets.sources import zeros_filter
from libs.us_state_abbrev import ABBREV_US_UNKNOWN_COUNTY_FIPS
from libs.datasets import tail_filter
from libs import pipeline
from libs.datasets.dataset_utils import DATA_DIRECTORY, AggregationLevel

import json
import logging

TailFilter = tail_filter.TailFilter

REGION_OVERRIDES_JSON = DATA_DIRECTORY / "region-overrides.json"

DATA_PATH_PREFIX = dataset_utils.DATA_DIRECTORY.relative_to(dataset_utils.REPO_ROOT)


KNOWN_LOCATION_ID_WITHOUT_POPULATION = [
    # Territories other than PR
    "iso1:us#iso2:us-vi",
    "iso1:us#iso2:us-as",
    "iso1:us#iso2:us-gu",
    # Subregion of AS
    "iso1:us#iso2:us-vi#fips:78030",
    "iso1:us#iso2:us-vi#fips:78020",
    "iso1:us#iso2:us-vi#fips:78010",
    # Retired FIPS
    "iso1:us#iso2:us-sd#fips:46113",
    "iso1:us#iso2:us-va#fips:51515",
    # All the unknown county FIPS
    *[pipeline.fips_to_location_id(f) for f in ABBREV_US_UNKNOWN_COUNTY_FIPS.values()],
]

CUMULATIVE_FIELDS_TO_FILTER = [
    CommonFields.CASES,
    CommonFields.DEATHS,
    CommonFields.POSITIVE_TESTS,
    CommonFields.NEGATIVE_TESTS,
    CommonFields.TOTAL_TESTS,
    CommonFields.POSITIVE_TESTS_VIRAL,
    CommonFields.POSITIVE_CASES_VIRAL,
    CommonFields.TOTAL_TESTS_VIRAL,
    CommonFields.TOTAL_TESTS_PEOPLE_VIRAL,
    CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL,
]
DEFAULT_REPORTING_RATIO = 0.95


@dataclass
class MultiRegionOrchestrator:

    regions: Iterable[MultiRegionDataset]  # Make a generator
    cbsa_aggregator: statistical_areas.CountyToCBSAAggregator
    region_overrides: Dict

    @classmethod
    def compute_and_persist_from_bulk_multiregion(
        self, states: Optional[List[str]] = None, refresh_datasets: Optional[bool] = True
    ) -> "MultiRegionOrchestrator":
        """Create an """
        bulk_dataset = load_bulk_dataset(refresh_datasets=refresh_datasets)
        if not states:
            states = list(US_STATE_ABBREV.values())
        regions = [bulk_dataset.get_subset(state=region) for region in states]

        cbsa_aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
        region_overrides = json.load(open(REGION_OVERRIDES_JSON))
        multiregion_dataset = MultiRegionOrchestrator(
            regions=regions, region_overrides=region_overrides, cbsa_aggregator=cbsa_aggregator
        ).build_and_combine_regions()

        logging.info("Writing multiregion_dataset...")
        combined_dataset_utils.persist_dataset(multiregion_dataset, DATA_PATH_PREFIX)
        logging.info("Finished multiregion_dataset")

    @staticmethod
    def update_single_state(state: str, print_stats: bool):
        """Recreate dataset for a specific state and its subregions leaving other regions unchanged"""
        # Never refresh datasets to ensure all data comes from the same parquet file.
        bulk_dataset = load_bulk_dataset(refresh_datasets=False)
        to_update, unchanged = bulk_dataset.partition_by_region(
            include=[
                pipeline.Region.from_state(state),
                pipeline.RegionMask(states=[state], level=AggregationLevel.COUNTY),
            ]
        )
        # TODO PARALLEL: check that metros are properly updated, too
        updated = _build_region_timeseries(to_update, print_stats=print_stats)
        multiregion_dataset = unchanged.append_regions(updated)
        combined_dataset_utils.persist_dataset(multiregion_dataset, DATA_PATH_PREFIX)

    def build_and_combine_regions(self):
        processed_regions = parallel_map(self._build_region_timeseries, self.regions)
        logging.info("Finished processing individual regions...")
        return combine_regions(list(processed_regions))

    def _build_region_timeseries(self, region_dataset: MultiRegionDataset, print_stats=True):
        region_overrides_config = manual_filter.transform_region_overrides(
            self.region_overrides, self.cbsa_aggregator.cbsa_to_counties_region_map
        )
        before_manual_filter = region_dataset
        multiregion_dataset = manual_filter.run(region_dataset, region_overrides_config)
        manual_filter_touched = manual_filter.touched_subset(
            before_manual_filter, multiregion_dataset
        )
        manual_filter_touched.write_to_wide_dates_csv(
            dataset_utils.MANUAL_FILTER_REMOVED_WIDE_DATES_CSV_PATH,
            dataset_utils.MANUAL_FILTER_REMOVED_STATIC_CSV_PATH,
        )
        if print_stats:
            region_dataset.print_stats("manual filter")
        multiregion_dataset = timeseries.drop_observations(
            region_dataset, after=datetime.datetime.utcnow().date()
        )

        multiregion_dataset = outlier_detection.drop_tail_positivity_outliers(multiregion_dataset)
        if print_stats:
            multiregion_dataset.print_stats(f"drop_tail {region_dataset.location_ids}")

        # Filter for stalled cumulative values before deriving NEW_CASES from CASES.
        _, multiregion_dataset = TailFilter.run(multiregion_dataset, CUMULATIVE_FIELDS_TO_FILTER)
        if print_stats:
            multiregion_dataset.print_stats("TailFilter")
        multiregion_dataset = zeros_filter.drop_all_zero_timeseries(
            multiregion_dataset,
            [
                CommonFields.VACCINES_DISTRIBUTED,
                CommonFields.VACCINES_ADMINISTERED,
                CommonFields.VACCINATIONS_COMPLETED,
                CommonFields.VACCINATIONS_INITIATED,
                CommonFields.VACCINATIONS_ADDITIONAL_DOSE,
            ],
        )
        if print_stats:
            multiregion_dataset.print_stats("zeros_filter")

        multiregion_dataset = vaccine_backfills.estimate_initiated_from_state_ratio(
            multiregion_dataset
        )
        if print_stats:
            multiregion_dataset.print_stats("estimate_initiated_from_state_ratio")

        multiregion_dataset = new_cases_and_deaths.add_new_cases(multiregion_dataset)
        multiregion_dataset = new_cases_and_deaths.add_new_deaths(multiregion_dataset)
        if print_stats:
            multiregion_dataset.print_stats("new_cases_and_deaths")

        multiregion_dataset = weekly_hospitalizations.add_weekly_hospitalizations(
            multiregion_dataset
        )

        multiregion_dataset = custom_patches.patch_maryland_missing_case_data(multiregion_dataset)
        if print_stats:
            multiregion_dataset.print_stats("patch_maryland_missing_case_data")

        multiregion_dataset = nytimes_anomalies.filter_by_nyt_anomalies(multiregion_dataset)
        if print_stats:
            multiregion_dataset.print_stats("nytimes_anomalies")

        multiregion_dataset = outlier_detection.drop_new_case_outliers(multiregion_dataset)
        multiregion_dataset = outlier_detection.drop_new_deaths_outliers(multiregion_dataset)
        if print_stats:
            multiregion_dataset.print_stats("outlier_detection")

        multiregion_dataset = timeseries.drop_regions_without_population(
            multiregion_dataset, KNOWN_LOCATION_ID_WITHOUT_POPULATION, structlog.get_logger()
        )
        if print_stats:
            multiregion_dataset.print_stats("drop_regions_without_population")

        multiregion_dataset = custom_aggregations.aggregate_puerto_rico_from_counties(
            multiregion_dataset
        )
        if print_stats:
            multiregion_dataset.print_stats("aggregate_puerto_rico_from_counties")
        multiregion_dataset = custom_aggregations.aggregate_to_new_york_city(multiregion_dataset)
        if print_stats:
            multiregion_dataset.print_stats("aggregate_to_new_york_city")

        # TODO: restrict HSA aggregation to create only rows for counties within state lines.
        # Or remove duplicates after the fact.
        hsa_aggregator = statistical_areas.CountyToHSAAggregator.from_local_data()
        multiregion_dataset = hsa_aggregator.aggregate(
            multiregion_dataset, restrict_to_current_state=True
        )
        if print_stats:
            multiregion_dataset.print_stats("CountyToHSAAggregator")

        multiregion_dataset = custom_aggregations.replace_dc_county_with_state_data(
            multiregion_dataset
        )
        if print_stats:
            multiregion_dataset.print_stats("replace_dc_county_with_state_data")
        return multiregion_dataset


def combine_regions(datasets: List[MultiRegionDataset]) -> MultiRegionDataset:
    # common_location_id = location_ids.intersection(other.location_ids)
    # if not common_location_id.empty:
    # raise ValueError("Do not use append_regions with duplicate location_id")
    logging.info("starting region combination...")
    timeseries_df = (
        pd.concat([dataset.timeseries_bucketed for dataset in datasets])
        .sort_index()
        .rename_axis(columns=PdFields.VARIABLE)
    )
    static_df = (
        pd.concat(dataset.static for dataset in datasets)
        .sort_index()
        .rename_axis(columns=PdFields.VARIABLE)
    )
    tag = pd.concat([dataset.tag for dataset in datasets]).sort_index()
    multiregion_dataset = MultiRegionDataset(
        timeseries_bucketed=timeseries_df, static=static_df, tag=tag
    )
    logging.info("starting CBSA aggregation...")
    aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
    cbsa_dataset = aggregator.aggregate(
        multiregion_dataset, reporting_ratio_required_to_aggregate=DEFAULT_REPORTING_RATIO
    )
    multiregion_dataset = multiregion_dataset.append_regions(cbsa_dataset)
    multiregion_dataset.print_stats("CBSA dataset")
    logging.info("starting country aggregation...")
    multiregion_dataset = custom_aggregations.aggregate_to_country(
        multiregion_dataset, reporting_ratio_required_to_aggregate=DEFAULT_REPORTING_RATIO
    )
    multiregion_dataset.print_stats("Aggregate to country")
    return multiregion_dataset


def load_bulk_dataset(refresh_datasets: Optional[bool] = True) -> MultiRegionDataset:
    if refresh_datasets:
        timeseries_field_datasets = load_datasets_by_field(ALL_TIMESERIES_FEATURE_DEFINITION)
        static_field_datasets = load_datasets_by_field(ALL_FIELDS_FEATURE_DEFINITION)

        multiregion_dataset = timeseries.combined_datasets(
            timeseries_field_datasets, static_field_datasets
        )
        multiregion_dataset.to_compressed_pickle(dataset_utils.COMBINED_RAW_PICKLE_GZ_PATH)
    else:
        multiregion_dataset = timeseries.MultiRegionDataset.from_compressed_pickle(
            dataset_utils.COMBINED_RAW_PICKLE_GZ_PATH
        )
    return multiregion_dataset


def load_datasets_by_field(
    feature_definition_config: combined_datasets.FeatureDataSourceMap, *, state=False, fips=False
) -> Mapping[FieldName, List[timeseries.MultiRegionDataset]]:
    def _load_dataset(data_source_cls) -> timeseries.MultiRegionDataset:
        try:
            dataset = data_source_cls.make_dataset()
            if state or fips:
                dataset = dataset.get_subset(state=state, fips=fips)
            return dataset
        except Exception:
            raise ValueError(f"Problem with {data_source_cls}")

    feature_definition = {
        # Put the highest priority first, as expected by timeseries.combined_datasets.
        # TODO(tom): reverse the hard-coded FeatureDataSourceMap and remove the reversed call.
        field_name: list(reversed(list(_load_dataset(cls) for cls in classes)))
        for field_name, classes in feature_definition_config.items()
        if classes
    }
    return feature_definition
