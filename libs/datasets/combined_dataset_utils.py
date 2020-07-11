from typing import Type, Tuple
import os
import tempfile
import pathlib
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
from libs.datasets.dataset_utils import DatasetPromotion
from libs.datasets.combined_dataset_pointer import CombinedDatasetPointer
from libs.datasets import combined_dataset_pointer
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
) -> CombinedDatasetPointer:

    model_git_info = GitSummary.from_repo_path(dataset_utils.REPO_ROOT)
    data_git_info = GitSummary.from_repo_path(data_public_path)

    if isinstance(dataset, timeseries.TimeseriesDataset):
        dataset_type = DatasetType.TIMESERIES
    elif isinstance(dataset, latest_values_dataset.LatestValuesDataset):
        dataset_type = DatasetType.LATEST

    filename = _form_dataset_filename(dataset_type, data_git_info, model_git_info)
    dataset_path = os.path.join(path_prefix, filename)
    dataset_pointer = CombinedDatasetPointer(
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
    pointer_path_dir: pathlib.Path = dataset_utils.POINTER_DIRECTORY,
    latest_dataset=None,
    timeseries_dataset=None,
):

    if not latest_dataset:
        latest_dataset = combined_datasets.build_us_latest_with_all_fields(skip_cache=True)
    latest_pointer = persist_dataset(latest_dataset, path_prefix)
    latest_pointer.save(pointer_path_dir, DatasetPromotion.LATEST)

    if not timeseries_dataset:
        timeseries_dataset = combined_datasets.build_timeseries_with_all_fields(skip_cache=True)
    timeseries_pointer = persist_dataset(timeseries_dataset, path_prefix)
    timeseries_pointer.save(pointer_path_dir, DatasetPromotion.LATEST)
    return latest_pointer, timeseries_pointer


def load_us_timeseries_with_all_fields(
    promotion_level: DatasetPromotion = DatasetPromotion.LATEST,
    pointer_directory: pathlib.Path = dataset_utils.POINTER_DIRECTORY,
    dataset_download_directory: pathlib.Path = dataset_utils.DATA_CACHE_FOLDER,
) -> timeseries.TimeseriesDataset:
    filename = combined_dataset_pointer.form_filename(DatasetType.TIMESERIES, promotion_level)
    pointer_path = pointer_directory / filename
    pointer = CombinedDatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(download_directory=dataset_download_directory)


def load_us_latest_with_all_fields(
    promotion_level: DatasetPromotion = DatasetPromotion.LATEST,
    pointer_directory: pathlib.Path = dataset_utils.POINTER_DIRECTORY,
    dataset_download_directory: pathlib.Path = dataset_utils.DATA_CACHE_FOLDER,
) -> latest_values_dataset.LatestValuesDataset:
    filename = combined_dataset_pointer.form_filename(DatasetType.LATEST, promotion_level)
    pointer_path = pointer_directory / filename
    pointer = CombinedDatasetPointer.parse_raw(pointer_path.read_text())
    return pointer.load_dataset(download_directory=dataset_download_directory)


def promote_pointer(
    dataset_type: DatasetType,
    from_level: DatasetPromotion,
    to_level: DatasetPromotion,
    pointer_directory: pathlib.Path = dataset_utils.POINTER_DIRECTORY,
):
    filename = combined_dataset_pointer.form_filename(dataset_type, from_level)
    pointer_path = pointer_directory / filename
    pointer = CombinedDatasetPointer.parse_raw(pointer_path.read_text())
    pointer.save(pointer_directory, to_level)
    _logger.info(
        "Successfully promoted dataset pointer", from_level=from_level.value, to=to_level.value
    )
