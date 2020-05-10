from typing import Dict, Type, List
import os
import tempfile
import pathlib
import logging
import time
import pandas as pd
from libs.datasets import dataset_base


PICKLE_CACHE_ENV_KEY = "PICKLE_CACHE_DIR"

_EXISTING_CACHE_KEYS = set()


_logger = logging.getLogger(__name__)


def set_pickle_cache_tempdir() -> str:
    """Sets the cache dir to a temporary directory.

    Note that the directory does not clean up after itself.
    """
    tempdir = tempfile.mkdtemp()
    os.environ[PICKLE_CACHE_ENV_KEY] = tempdir
    _logger.info(f"Setting pickle cache tmpdir to {tempdir}")
    return tempdir


def cache_dataset_on_disk(
    target_dataset_cls: Type[dataset_base.DatasetBase], max_age_in_minutes=30, key=None
):
    """Caches underlying pandas data from to an on disk location.

    Args:
        target_dataset_cls: Class of dataset to wrap pandas data with.
        max_age_in_minutes: Maximum age of cache before it becomes stale.
        key: Cache key. If not specified, uses name of function.
    """

    def decorator(func):
        cache_key = key or func.__name__
        print(cache_key)
        if cache_key in _EXISTING_CACHE_KEYS:
            raise ValueError(
                f"Have already wrapped a function with the key name: {func.__name__}. "
                "Please specify a different key."
            )
        _EXISTING_CACHE_KEYS.add(cache_key)

        def f() -> target_dataset_cls:
            pickle_cache_dir = os.getenv(PICKLE_CACHE_ENV_KEY)
            if not pickle_cache_dir:
                return func()

            cache_path = pathlib.Path(pickle_cache_dir) / (cache_key + ".pickle")

            if cache_path.exists():
                modified_time = cache_path.stat().st_mtime
                cache_age_in_minutes = (time.time() - modified_time) / 60
                if cache_age_in_minutes < max_age_in_minutes:
                    return target_dataset_cls(pd.read_pickle(cache_path))
                else:
                    _logger.debug(f"Cache expired, reloading.")

            dataset = func()
            dataset.data.to_pickle(cache_path)
            return dataset

        return f

    return decorator
