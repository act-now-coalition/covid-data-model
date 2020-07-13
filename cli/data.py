import logging
import pathlib
import click
from libs.datasets import dataset_utils
from libs.datasets import combined_dataset_utils
from libs.datasets.combined_dataset_utils import DatasetTag
from libs.datasets.combined_dataset_utils import DatasetType


PROD_BUCKET = "data.covidactnow.org"

_logger = logging.getLogger(__name__)


@click.group("data")
def main():
    pass


@main.command()
def update_latest():
    """Updates latest and timeseries datasets to the current checked out covid data public commit"""
    path_prefix = dataset_utils.DATA_CACHE_FOLDER.relative_to(dataset_utils.REPO_ROOT)
    combined_dataset_utils.update_data_public_head(path_prefix)


@main.command()
@click.option("--cleanup", is_flag=True, help="If set, removes datasets not referenced by pointers")
def promote_latest(cleanup: bool):
    """Promotes latest and timeseries datasets to stable."""
    combined_dataset_utils.promote_pointer(DatasetType.LATEST, DatasetTag.LATEST, DatasetTag.STABLE)
    combined_dataset_utils.promote_pointer(
        DatasetType.TIMESERIES, DatasetTag.LATEST, DatasetTag.STABLE
    )
    if cleanup:
        combined_dataset_utils.remove_stale_datasets(
            dataset_utils.POINTER_DIRECTORY, dataset_utils.DATA_CACHE_FOLDER
        )


@main.command()
@click.option(
    "--pointer-dir",
    type=pathlib.Path,
    help="Path to dataset pointer directory.",
    default=dataset_utils.POINTER_DIRECTORY,
)
@click.option(
    "--dataset-cache-dir",
    type=pathlib.Path,
    help="Path to dataset cache directory.",
    default=dataset_utils.DATA_CACHE_FOLDER,
)
@click.option("--dry-run", is_flag=True)
def cleanup_stale(pointer_dir: pathlib.Path, dataset_cache_dir: pathlib.Path, dry_run: bool):
    combined_dataset_utils.remove_stale_datasets(pointer_dir, dataset_cache_dir, dry_run=dry_run)
