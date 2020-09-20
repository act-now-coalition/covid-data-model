from itertools import chain
from typing import Optional
import logging
import pathlib
import os
import json
import shutil

import click

from libs import google_sheet_helpers, wide_dates_df
from libs.datasets.combined_datasets import (
    ALL_TIMESERIES_FEATURE_DEFINITION,
    US_STATES_FILTER,
    ALL_FIELDS_FEATURE_DEFINITION,
)
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.qa import dataset_summary
from libs.qa import data_availability
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import dataset_utils
from libs.datasets import combined_dataset_utils
from libs.datasets import combined_datasets
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.sources import forecast_hub
from pyseir import DATA_DIR
import pyseir.icu.utils
from pyseir.icu import infer_icu


PROD_BUCKET = "data.covidactnow.org"

_logger = logging.getLogger(__name__)


@click.group("data")
def main():
    pass


def _save_field_summary(
    timeseries_dataset: MultiRegionTimeseriesDataset, output_path: pathlib.Path
):

    _logger.info("Starting dataset summary generation")
    summary = dataset_summary.summarize_timeseries_fields(timeseries_dataset.data)
    summary.to_csv(output_path)
    _logger.info(f"Saved dataset summary to {output_path}")


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
@click.option("--summary-filename", default="timeseries_summary.csv")
@click.option("--wide-dates-filename", default="timeseries-wide-dates.csv")
def update(summary_filename, wide_dates_filename):
    """Updates latest and timeseries datasets to the current checked out covid data public commit"""
    path_prefix = dataset_utils.DATA_DIRECTORY.relative_to(dataset_utils.REPO_ROOT)

    data_source_classes = set(
        chain(
            chain.from_iterable(ALL_FIELDS_FEATURE_DEFINITION.values()),
            chain.from_iterable(ALL_TIMESERIES_FEATURE_DEFINITION.values()),
        )
    )
    data_sources = {
        data_source_cls.SOURCE_NAME: data_source_cls.local()
        for data_source_cls in data_source_classes
    }
    timeseries_dataset: TimeseriesDataset = combined_datasets.build_from_sources(
        TimeseriesDataset, data_sources, ALL_TIMESERIES_FEATURE_DEFINITION, filter=US_STATES_FILTER
    )
    multiregion_dataset = MultiRegionTimeseriesDataset.from_timeseries(timeseries_dataset)
    latest_dataset: LatestValuesDataset = combined_datasets.build_from_sources(
        LatestValuesDataset, data_sources, ALL_FIELDS_FEATURE_DEFINITION, filter=US_STATES_FILTER,
    )
    _, timeseries_pointer = combined_dataset_utils.update_data_public_head(
        path_prefix, latest_dataset, multiregion_dataset,
    )

    # Write DataSource objects that have provenance information, which is only set when significant
    # processing of the source data is done in this repo before it is combined. The output is not
    # used downstream, it is for debugging only.
    for data_source in data_sources.values():
        if data_source.provenance is not None:
            wide_dates_df.write_csv(
                data_source.timeseries().get_date_columns(),
                path_prefix / f"{data_source.SOURCE_NAME}-wide-dates.csv",
            )

    if wide_dates_filename:
        wide_dates_df.write_csv(
            timeseries_dataset.get_date_columns(),
            timeseries_pointer.path.with_name(wide_dates_filename),
        )

    if summary_filename:
        _save_field_summary(multiregion_dataset, path_prefix / summary_filename)


@main.command()
@click.option("--output-dir", type=pathlib.Path, required=True)
@click.option("--filename", type=pathlib.Path, default="timeseries_field_summary.csv")
@click.option("--level", type=AggregationLevel)
def save_summary(output_dir: pathlib.Path, filename: str, level: Optional[AggregationLevel]):
    """Saves summary of timeseries dataset indexed by fips and variable name."""

    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    if level:
        us_timeseries = MultiRegionTimeseriesDataset.from_timeseries(
            us_timeseries.to_timeseries().get_subset(aggregation_level=level)
        )

    _save_field_summary(us_timeseries, output_dir / filename)


@main.command()
@click.option("--name", envvar="DATA_AVAILABILITY_SHEET_NAME", default="Data Availability - Dev")
@click.option("--share-email")
def update_availability_report(name: str, share_email: Optional[str]):
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
        json.dump(output, f)
