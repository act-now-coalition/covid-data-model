import logging
import click
from libs.datasets import combined_dataset_utils
from libs.datasets.combined_dataset_utils import DatasetPromotion
from libs.datasets.combined_dataset_utils import DatasetType


PROD_BUCKET = "data.covidactnow.org"

_logger = logging.getLogger(__name__)


@click.group("data")
def main():
    pass


@main.command()
def update_latest():
    """Updates latest and timeseries datasets to the current checked out covid data public commit"""
    combined_dataset_utils.update_data_public_head(combined_dataset_utils.DATA_CACHE_FOLDER)


@main.command()
def promote_latest():
    """Promotes latest and timeseries datasets to stable."""
    combined_dataset_utils.promote_pointer(
        DatasetType.LATEST, DatasetPromotion.LATEST, DatasetPromotion.STABLE
    )
    combined_dataset_utils.promote_pointer(
        DatasetType.TIMESERIES, DatasetPromotion.LATEST, DatasetPromotion.STABLE
    )
