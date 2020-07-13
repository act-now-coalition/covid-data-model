import logging
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
def promote_latest():
    """Promotes latest and timeseries datasets to stable."""
    combined_dataset_utils.promote_pointer(DatasetType.LATEST, DatasetTag.LATEST, DatasetTag.STABLE)
    combined_dataset_utils.promote_pointer(
        DatasetType.TIMESERIES, DatasetTag.LATEST, DatasetTag.STABLE
    )
