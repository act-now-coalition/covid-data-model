from itertools import chain
from typing import Optional
import logging
import pathlib
import os
import json
import shutil
import structlog

import click
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields

from libs import google_sheet_helpers, wide_dates_df
from libs import pipeline
from libs.datasets import AggregationLevel
from libs.datasets import combined_dataset_utils
from libs.datasets import custom_aggregations
from libs.datasets import statistical_areas
from libs.datasets.combined_datasets import (
    ALL_TIMESERIES_FEATURE_DEFINITION,
    ALL_FIELDS_FEATURE_DEFINITION,
)
from libs.datasets.timeseries import DatasetName
from libs.datasets import timeseries
from libs.datasets import dataset_utils
from libs.datasets import combined_datasets
from libs.datasets.sources import forecast_hub
from libs.us_state_abbrev import ABBREV_US_UNKNOWN_COUNTY_FIPS
from pyseir import DATA_DIR
import pyseir.icu.utils
from pyseir.icu import infer_icu


PROD_BUCKET = "data.covidactnow.org"

_logger = logging.getLogger(__name__)


@click.group("data")
def main():
    pass


@main.command()
@click.option("--filename", default="external_forecasts.csv")
def update_forecasts(filename):
    """Updates external forecasts to the current checked out covid data public commit"""
    path_prefix = dataset_utils.DATA_DIRECTORY.relative_to(dataset_utils.REPO_ROOT)
    data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
    data_path = forecast_hub.ForecastHubDataset.DATA_PATH
    shutil.copy(data_root / data_path, path_prefix / filename)
    _logger.info(f"Updating External Forecasts at {path_prefix / filename}")


@main.command()
@click.option("--wide-dates-filename", default="multiregion-wide-dates.csv")
@click.option(
    "--aggregate-to-country/--no-aggregate-to-country",
    is_flag=True,
    help="Aggregate states to one USA country region",
    default=False,
)
@click.option("--state", type=str, help="For testing, a two letter state abbr")
@click.option("--fips", type=str, help="For testing, a 5 digit county fips")
def update(
    wide_dates_filename, aggregate_to_country: bool, state: Optional[str], fips: Optional[str]
):
    """Updates latest and timeseries datasets to the current checked out covid data public commit"""
    path_prefix = dataset_utils.DATA_DIRECTORY.relative_to(dataset_utils.REPO_ROOT)

    data_source_classes = set(
        chain(
            chain.from_iterable(ALL_FIELDS_FEATURE_DEFINITION.values()),
            chain.from_iterable(ALL_TIMESERIES_FEATURE_DEFINITION.values()),
        )
    )
    data_sources = {
        data_source_cls.SOURCE_NAME: data_source_cls.local().multi_region_dataset()
        for data_source_cls in data_source_classes
    }
    if state or fips:
        data_sources = {
            name: dataset.get_subset(state=state, fips=fips)
            # aggregation_level=AggregationLevel.STATE)
            for name, dataset in data_sources.items()
        }
    multiregion_dataset = timeseries.combined_datasets(
        data_sources,
        build_field_dataset_source(ALL_TIMESERIES_FEATURE_DEFINITION),
        build_field_dataset_source(ALL_FIELDS_FEATURE_DEFINITION),
    )
    multiregion_dataset = timeseries.add_new_cases(multiregion_dataset)
    multiregion_dataset = timeseries.drop_new_case_outliers(multiregion_dataset)
    multiregion_dataset = timeseries.drop_regions_without_population(
        multiregion_dataset, KNOWN_LOCATION_ID_WITHOUT_POPULATION, structlog.get_logger()
    )
    multiregion_dataset = timeseries.aggregate_puerto_rico_from_counties(multiregion_dataset)
    multiregion_dataset = custom_aggregations.aggregate_to_new_york_city(multiregion_dataset)

    aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
    cbsa_dataset = aggregator.aggregate(multiregion_dataset)
    multiregion_dataset = multiregion_dataset.append_regions(cbsa_dataset)

    if aggregate_to_country:
        country_dataset = timeseries.aggregate_regions(
            multiregion_dataset, pipeline.us_states_to_country_map(), AggregationLevel.COUNTRY
        )
        multiregion_dataset = multiregion_dataset.append_regions(country_dataset)

    combined_dataset_utils.persist_dataset(multiregion_dataset, path_prefix)

    # Write DataSource objects that have provenance information, which is only set when significant
    # processing of the source data is done in this repo before it is combined. The output is not
    # used downstream, it is for debugging only.
    for name, source_dataset in data_sources.items():
        if not source_dataset.provenance.empty:
            wide_dates_df.write_csv(
                source_dataset.timeseries_rows(), path_prefix / f"{name}-wide-dates.csv",
            )

    if wide_dates_filename:
        wide_dates_df.write_csv(
            multiregion_dataset.timeseries_rows(), path_prefix / wide_dates_filename,
        )
        static_sorted = common_df.index_and_sort(
            multiregion_dataset.static,
            index_names=[CommonFields.LOCATION_ID],
            log=structlog.get_logger(),
        )
        static_sorted.to_csv(path_prefix / wide_dates_filename.replace("wide-dates", "static"))


@main.command()
@click.argument("output_path", type=pathlib.Path)
def aggregate_cbsa(output_path: pathlib.Path):
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
    cbsa_dataset = aggregator.aggregate(us_timeseries)
    cbsa_dataset.to_csv(output_path)


@main.command()
@click.argument("output_path", type=pathlib.Path)
def aggregate_states_to_country(output_path: pathlib.Path):
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    country_dataset = timeseries.aggregate_regions(
        us_timeseries, pipeline.us_states_to_country_map(), AggregationLevel.COUNTRY
    )
    country_dataset.to_csv(output_path)


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


@main.command()
@click.argument("output_path", type=pathlib.Path)
def run_population_filter(output_path: pathlib.Path):
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    log = structlog.get_logger()
    log.info("starting filter")
    ts_out = timeseries.drop_regions_without_population(
        us_timeseries, KNOWN_LOCATION_ID_WITHOUT_POPULATION, log
    )
    ts_out.to_csv(output_path)


@main.command()
@click.option("--name", envvar="DATA_AVAILABILITY_SHEET_NAME", default="Data Availability - Dev")
@click.option("--share-email")
def update_availability_report(name: str, share_email: Optional[str]):
    from libs.qa import data_availability

    sheet = google_sheet_helpers.open_or_create_spreadsheet(name, share_email=share_email)
    info_worksheet = google_sheet_helpers.update_info_sheet(sheet)
    data_sources_by_source_name = data_availability.load_all_latest_sources()

    for name, dataset in data_sources_by_source_name.items():
        _logger.info(f"Updating {name}")
        report = data_availability.build_data_availability_report(dataset)
        data_availability.update_multi_field_availability_report(
            sheet, report, name, columns_to_drop=["source", "fips"]
        )

    # Reorder sheets with combined data first and metadata last
    COLUMN_ORDER_OVERRIDE = {data_availability.COMBINED_DATA_KEY: -5, info_worksheet.title: 5}
    worksheets = sheet.worksheets()
    worksheets = sorted(worksheets, key=lambda x: (COLUMN_ORDER_OVERRIDE.get(x.title, 0), x.title))
    sheet.reorder_worksheets(worksheets)

    _logger.info("Finished updating data availability report")


@main.command()
def update_case_based_icu_utilization_weights():
    """
    Calculate the updated States to Counties disaggregation weights and save to disk. These
    weights are used to estimate county level ICU heads-in-beds as an input for the ICU Utilization
    metric.

    The output is callable with county aggregation-level fips keys and returns a normalized [0,1]
    value such that the weights for all counties in a given state sum to unity.
    """
    output_path = os.path.join(DATA_DIR, infer_icu.ICUWeightsPath.ONE_MONTH_TRAILING_CASES.value)
    output = pyseir.icu.utils.calculate_case_based_weights()
    _logger.info(f"Saved case-based ICU Utilization weights to {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, sort_keys=True)


def build_field_dataset_source(feature_definition_config):
    feature_definition = {
        # timeseries.combined_datasets has the highest priority first.
        # TODO(tom): reverse the hard-coded FeatureDataSourceMap and remove the reversed call.
        field_name: list(reversed(list(DatasetName(cls.SOURCE_NAME) for cls in classes)))
        for field_name, classes in feature_definition_config.items()
        if classes
    }
    return feature_definition
