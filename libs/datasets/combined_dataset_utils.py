from typing import Type, Tuple
import os
import tempfile
import pathlib
import functools
import datetime
import enum
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
from libs.datasets.dataset_utils import DatasetType
from libs.datasets.dataset_utils import DatasetTag
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets import dataset_pointer
from libs.github_utils import GitSummary

_logger = structlog.getLogger(__name__)


def _form_dataset_filename(
    dataset_type: DatasetType, data_git_info: GitSummary, model_git_info: GitSummary
) -> str:
    path_format = "{dataset_type}.{timestamp}.{model_sha}-{data_sha}.csv"
    return path_format.format(
        dataset_type=dataset_type.value,
        data_sha=data_git_info.sha[:8],
        model_sha=model_git_info.sha[:8],
        timestamp=datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
    )


def persist_dataset(
    dataset: dataset_base.DatasetBase,
    path_prefix: str,
    data_public_path: pathlib.Path = dataset_utils.LOCAL_PUBLIC_DATA_PATH,
    s3_client=None,
) -> DatasetPointer:
    """Creates a DatasetPointer and persists dataset to directory of path_prefix.

    Can save dataset locally or to s3.

    Args:
        dataset: Dataset to persist.
        path_prefix: Path prefix of dataset. Can either be an s3 url
            (i.e. "s3://<bucket>/<subfolder>") or a local path.
        data_public_path: Path to covid data public folder.
        s3_client: Optional s3 client.

    Returns: DatasetPointer describing persisted dataset.
    """
    model_git_info = GitSummary.from_repo_path(dataset_utils.REPO_ROOT)
    data_git_info = GitSummary.from_repo_path(data_public_path)

    if isinstance(dataset, timeseries.TimeseriesDataset):
        dataset_type = DatasetType.TIMESERIES
    elif isinstance(dataset, latest_values_dataset.LatestValuesDataset):
        dataset_type = DatasetType.LATEST

    filename = _form_dataset_filename(dataset_type, data_git_info, model_git_info)
    dataset_path = os.path.join(path_prefix, filename)
    dataset_pointer = DatasetPointer(
        dataset_type=dataset_type,
        path=dataset_path,
        data_git_info=data_git_info,
        model_git_info=model_git_info,
        updated_at=datetime.datetime.utcnow(),
    )
    if dataset_pointer.is_s3:
        s3_client = s3_client or boto3.client("s3")
        dataset_pointer.upload_dataset(dataset, s3_client)
    else:
        dataset_pointer.save_dataset(dataset)

    return dataset_pointer


def update_data_public_head(
    path_prefix: str,
    tag: DatasetTag = DatasetTag.LATEST,
    pointer_path_dir: pathlib.Path = dataset_utils.POINTER_DIRECTORY,
    latest_dataset: latest_values_dataset.LatestValuesDataset = None,
    timeseries_dataset: timeseries.TimeseriesDataset = None,
) -> Tuple[DatasetPointer, DatasetPointer]:
    """Persists US latest and timeseries dataset and saves dataset pointers for Latest tag.

    Args:
        path_prefix: Path prefix of dataset. Can either be an s3 url
            (i.e. "s3://<bucket>/<subfolder>") or a local path.
        tag: DatasetTag to save pointer as.
        pointer_path_dir: Directory to save DatasetPointer files.
        latest_dataset: Optionally specify a LatestValuesDataset to persist instead of building
            from head.  Generally used in testing to sidestep building entire dataset.
        timeseries_dataset: Optionally specify a TimeseriesDataset to persist instead of building
            from head.  Generally used in testing to sidestep building entire dataset.

    Returns: Tuple of DatasetPointers to latest and timeseries datasets.
    """
    if not latest_dataset:
        latest_dataset = combined_datasets.load_us_latest_dataset(skip_cache=True)
    latest_pointer = persist_dataset(latest_dataset, path_prefix)
    latest_pointer.save(pointer_path_dir, tag)

    if not timeseries_dataset:
        timeseries_dataset = combined_datasets.load_us_timeseries_dataset(skip_cache=True)
    timeseries_pointer = persist_dataset(timeseries_dataset, path_prefix)
    timeseries_pointer.save(pointer_path_dir, tag)
    return latest_pointer, timeseries_pointer


def promote_pointer(
    dataset_type: DatasetType,
    from_tag: DatasetTag,
    to_tag: DatasetTag,
    pointer_directory: pathlib.Path = dataset_utils.POINTER_DIRECTORY,
):
    filename = dataset_pointer.form_filename(dataset_type, from_tag)
    pointer_path = pointer_directory / filename
    pointer = DatasetPointer.parse_raw(pointer_path.read_text())
    pointer.save(pointer_directory, to_tag)
    _logger.info("Successfully promoted dataset pointer", from_tag=from_tag.value, to=to_tag.value)
