from typing import Type, Tuple
import os
import tempfile
import pathlib
import datetime
from urllib.parse import urlparse

import boto3
import structlog
import numpy as np
import pandas as pd
import pydantic
from covidactnow.datapublic import common_df

from libs.datasets import dataset_base
from libs.datasets import combined_datasets
from libs.datasets import timeseries
from libs.datasets import latest_values_dataset
from libs.datasets import dataset_utils
from libs.datasets.dataset_base import DatasetBase
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.dataset_utils import DatasetTag
from libs.github_utils import GitSummary

_logger = structlog.getLogger(__name__)


def s3_split(url) -> Tuple[str, str]:
    """Split S3 URL into bucket and key.

    Args:
        url: S3 URL.

    Returns: Tuple of bucket, key
    """
    results = urlparse(url, allow_fragments=False)
    return results.netloc, results.path.lstrip("/")


def form_filename(dataset_type: DatasetType, dataset_tag: DatasetTag) -> str:
    return f"{dataset_type.value}.{dataset_tag.value}.json"


class CombinedDatasetPointer(pydantic.BaseModel):
    """Describes a persisted combined dataset."""

    dataset_type: DatasetType

    # Can be either an s3 url or local path.
    path: str

    # Sha of covid-data-public for dataset
    data_git_info: GitSummary

    # Sha of covid-data-model used to create dataset.
    model_git_info: GitSummary

    # When local file was saved.
    updated_at: datetime.datetime

    @property
    def is_s3(self):
        return self.path.startswith("s3://")

    @property
    def filename(self) -> str:
        *_, filename = os.path.split(self.path)
        return filename

    def download(
        self,
        directory: pathlib.Path = dataset_utils.DATA_CACHE_FOLDER,
        overwrite=False,
        s3_client=None,
    ) -> pathlib.Path:
        if not self.is_s3:
            return pathlib.Path(self.path)

        s3_client = s3_client or boto3.client("s3")
        dest_path = directory / self.filename
        if dest_path.exists() and not overwrite:
            raise FileExistsError()

        bucket, key = s3_split(self.path)
        s3_client.download_file(bucket, key, str(dest_path))
        return dest_path

    def upload_dataset(self, dataset: DatasetBase, s3_client):
        if not self.is_s3:
            raise Exception("Cannot only upload datasets if path is an s3 url.")

        with tempfile.NamedTemporaryFile() as tmp_file:
            path = pathlib.Path(tmp_file.name)
            dataset.to_csv(path)
            bucket, key = s3_split(self.path)
            s3_client.upload_file(str(path), bucket, key)
            _logger.info("Successfully uploaded dataset", path=self.path)

    def save_dataset(self, dataset: DatasetBase):
        dataset.to_csv(pathlib.Path(self.path))
        _logger.info("Successfully saved dataset", path=self.path)

    def load_dataset(self, download_directory: pathlib.Path = dataset_utils.DATA_CACHE_FOLDER):
        if not self.is_s3:
            return self.dataset_type.dataset_class.load_csv(self.path)

        path = download_directory / self.filename
        if not path.exists():
            path = self.download(directory=download_directory)

        return self.dataset_type.dataset_class.load_csv(path)

    def save(self, directory: pathlib.Path, dataset_tag: DatasetTag) -> pathlib.Path:
        filename = form_filename(self.dataset_type, dataset_tag)
        path = directory / filename
        path.write_text(self.json(indent=2))
        return path
