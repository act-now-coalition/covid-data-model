from typing import Type
import functools
import os
import tempfile
import pathlib
import logging
import time
import pandas as pd
from libs.datasets import dataset_base

# Defaulting cache dir to pyseir_data folder for convenience.
# Note: this means that in development, data will be cached by default to
# pyseir_data.
DEFAULT_CACHE_DIR = "pyseir_data"
PICKLE_CACHE_ENV_KEY = "PICKLE_CACHE_DIR"

_EXISTING_CACHE_KEYS = set()


_logger = logging.getLogger(__name__)


def set_pickle_cache_dir(force=False, cache_dir=DEFAULT_CACHE_DIR) -> str:
    """Sets the cache dir to default cache directory.

    Note that the directory does not clean up after itself.

    Args:
        force: If True, will force a cache key to be a new tempdir
            if key already exists.
        cache_dir: default directory, if None will set to a temporary directory.
    """

    if os.getenv(PICKLE_CACHE_ENV_KEY) and not force:
        directory = os.getenv(PICKLE_CACHE_ENV_KEY)
        _logger.info(f"Using existing pickle cache tmpdir: {directory}")
        return directory

    cache_dir = cache_dir or tempfile.mkdtemp()
    os.environ[PICKLE_CACHE_ENV_KEY] = cache_dir
    _logger.info(f"Setting {PICKLE_CACHE_ENV_KEY} to {cache_dir}.")
    return cache_dir


def cache_dataset_on_disk(
    target_dataset_cls: Type[dataset_base.DatasetBase], max_age_in_minutes=240, key=None
):
    """Caches underlying pandas data from to an on disk location.

    Args:
        target_dataset_cls: Class of dataset to wrap pandas data with.
        max_age_in_minutes: Maximum age of cache before it becomes stale.
        key: Cache key. If not specified, uses name of function.
    """

    def decorator(func):
        cache_key = key or func.__name__

        # Don't raise an error if the cache dir isn't set to prevent errors when developing
        # locally and running code in jupyter notebooks with autoreload set to true.
        if os.getenv(PICKLE_CACHE_ENV_KEY) and cache_key in _EXISTING_CACHE_KEYS:
            raise ValueError(
                f"Have already wrapped a function with the key name: {func.__name__}. "
                "Please specify a different key."
            )
        _EXISTING_CACHE_KEYS.add(cache_key)

        # load cache once per decorator.  If cache dir is not set, this will always be None.
        _loaded_cache = None

        @functools.wraps(func)
        def f() -> target_dataset_cls:
            nonlocal _loaded_cache

            pickle_cache_dir = os.getenv(PICKLE_CACHE_ENV_KEY)

            if not pickle_cache_dir:
                return func()

            if _loaded_cache is not None:
                return _loaded_cache

            cache_path = pathlib.Path(pickle_cache_dir) / (cache_key + ".pickle")

            if cache_path.exists():
                modified_time = cache_path.stat().st_mtime
                cache_age_in_minutes = (time.time() - modified_time) / 60
                if cache_age_in_minutes < max_age_in_minutes:
                    _logger.info(f"Loading {func.__name__} from on disk cache at {cache_path}")
                    _loaded_cache = target_dataset_cls(pd.read_pickle(cache_path))
                    return _loaded_cache
                else:
                    _logger.info("Cache expired, reloading.")

            dataset = func()
            dataset.data.to_pickle(cache_path)
            _loaded_cache = dataset
            return dataset

        return f

    return decorator
