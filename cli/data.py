import logging
import click
from libs.datasets import combined_dataset_utils


PROD_BUCKET = "data.covidactnow.org"

_logger = logging.getLogger(__name__)


@click.group("data")
def main():
    pass


@main.command()
def update_latest():
    combined_dataset_utils.update_data_public_head(combined_dataset_utils.DATA_CACHE_FOLDER,)
