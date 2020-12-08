import pathlib
import datetime
import structlog

from libs.datasets import dataset_base
from libs.datasets import dataset_utils
from libs.datasets.dataset_pointer import DatasetPointer
from libs.github_utils import GitSummary

_logger = structlog.getLogger(__name__)


def persist_dataset(
    dataset: dataset_base.SaveableDatasetInterface,
    data_directory: pathlib.Path,
    data_public_path: pathlib.Path = dataset_utils.LOCAL_PUBLIC_DATA_PATH,
) -> DatasetPointer:
    """Saves dataset and associated pointer in same data directory.

    Args:
        dataset: Dataset to persist.
        data_directory: Data directory
        data_public_path: Path to covid data public folder.

    Returns: DatasetPointer describing persisted dataset.
    """
    model_git_info = GitSummary.from_repo_path(dataset_utils.REPO_ROOT)
    data_git_info = GitSummary.from_repo_path(data_public_path)

    dataset_type = dataset.dataset_type

    dataset_path = data_directory / f"{dataset_type.value}.csv"
    dataset_pointer = DatasetPointer(
        dataset_type=dataset_type,
        path=dataset_path,
        data_git_info=data_git_info,
        model_git_info=model_git_info,
        updated_at=datetime.datetime.utcnow(),
    )
    dataset_pointer.save_dataset(dataset)
    dataset_pointer.save(data_directory)
    return dataset_pointer
