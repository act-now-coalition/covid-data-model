from typing import Optional
import logging
import pathlib
import click
from libs.qa import dataset_summary
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import dataset_utils
from libs.datasets import combined_dataset_utils
from libs.datasets import combined_datasets
from libs.datasets.combined_dataset_utils import DatasetType
from libs.datasets.dataset_utils import AggregationLevel


PROD_BUCKET = "data.covidactnow.org"

_logger = logging.getLogger(__name__)


@click.group("data")
def main():
    pass


def _save_field_summary(timeseries_dataset: TimeseriesDataset, output_path: pathlib.Path):

    _logger.info("Starting dataset summary generation")
    summary = dataset_summary.summarize_timeseries_fields(timeseries_dataset.data)
    summary.to_csv(output_path)
    _logger.info(f"Saved dataset summary to {output_path}")


@main.command()
@click.option("--summary-filename", default="timeseries_summary.csv")
def update(summary_filename):
    """Updates latest and timeseries datasets to the current checked out covid data public commit"""
    path_prefix = dataset_utils.DATA_DIRECTORY.relative_to(dataset_utils.REPO_ROOT)
    _, timeseries_pointer = combined_dataset_utils.update_data_public_head(path_prefix)

    dataset = timeseries_pointer.load_dataset()
    _save_field_summary(dataset, path_prefix / summary_filename)


@main.command()
@click.option("--output-dir", type=pathlib.Path, required=True)
@click.option("--filename", type=pathlib.Path, default="timeseries_field_summary.csv")
@click.option("--level", type=AggregationLevel)
def save_summary(output_dir: pathlib.Path, filename: str, level: Optional[AggregationLevel]):
    """Saves summary of timeseries dataset indexed by fips and variable name."""

    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    if level:
        us_timeseries = us_timeseries.get_subset(aggregation_level=level)

    _save_field_summary(us_timeseries, output_dir / filename)
