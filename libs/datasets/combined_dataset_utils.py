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
from libs.datasets.dataset_pointer import DatasetPointer
from libs.datasets import dataset_pointer
from libs.github_utils import GitSummary

_logger = structlog.getLogger(__name__)


def persist_dataset(
    dataset: dataset_base.DatasetBase,
    path_prefix: pathlib.Path,
    data_public_path: pathlib.Path = dataset_utils.LOCAL_PUBLIC_DATA_PATH,
) -> DatasetPointer:
    """Creates a DatasetPointer and persists dataset to directory of path_prefix.

    Can save dataset locally or to s3.

    Args:
        dataset: Dataset to persist.
        path_prefix: Path prefix of dataset.
        data_public_path: Path to covid data public folder.

    Returns: DatasetPointer describing persisted dataset.
    """
    model_git_info = GitSummary.from_repo_path(dataset_utils.REPO_ROOT)
    data_git_info = GitSummary.from_repo_path(data_public_path)

    if isinstance(dataset, timeseries.TimeseriesDataset):
        dataset_type = DatasetType.TIMESERIES
    elif isinstance(dataset, latest_values_dataset.LatestValuesDataset):
        dataset_type = DatasetType.LATEST

    dataset_path = path_prefix / f"{dataset_type.value}.csv"
    dataset_pointer = DatasetPointer(
        dataset_type=dataset_type,
        path=dataset_path,
        data_git_info=data_git_info,
        model_git_info=model_git_info,
        updated_at=datetime.datetime.utcnow(),
    )
    dataset_pointer.save_dataset(dataset)
    return dataset_pointer


def update_data_public_head(
    path_prefix: pathlib.Path,
    pointer_path_dir: pathlib.Path = dataset_utils.POINTER_DIRECTORY,
    latest_dataset: latest_values_dataset.LatestValuesDataset = None,
    timeseries_dataset: timeseries.TimeseriesDataset = None,
) -> Tuple[DatasetPointer, DatasetPointer]:
    """Persists US latest and timeseries dataset and saves dataset pointers for Latest tag.

    Args:
        path_prefix: Path prefix of dataset
        pointer_path_dir: Directory to save DatasetPointer files.
        latest_dataset: Optionally specify a LatestValuesDataset to persist instead of building
            from head.  Generally used in testing to sidestep building entire dataset.
        timeseries_dataset: Optionally specify a TimeseriesDataset to persist instead of building
            from head.  Generally used in testing to sidestep building entire dataset.

    Returns: Tuple of DatasetPointers to latest and timeseries datasets.
    """
    if not latest_dataset:
        latest_dataset = combined_datasets.build_us_latest_with_all_fields(skip_cache=True)
    latest_pointer = persist_dataset(latest_dataset, path_prefix)
    latest_pointer.save(pointer_path_dir)

    if not timeseries_dataset:
        timeseries_dataset = combined_datasets.build_us_timeseries_with_all_fields(skip_cache=True)
    timeseries_pointer = persist_dataset(timeseries_dataset, path_prefix)
    timeseries_pointer.save(pointer_path_dir)
    return latest_pointer, timeseries_pointer
