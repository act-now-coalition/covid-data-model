import os
import platform
from multiprocessing import get_context
from typing import Callable, TypeVar, Iterable

from pandarallel import pandarallel
import pandas as pd
import structlog

_log = structlog.get_logger()

VISIBIBLE_PROGRESS_BAR = os.environ.get("PYSEIR_VERBOSITY") == "True"
pandarallel.initialize(progress_bar=VISIBIBLE_PROGRESS_BAR)

FORCE_MULTIPROCESSING = str(os.environ.get("FORCE_MULTIPROCESSING")).lower() in ["true", "1"]

# multiprocessing is unreliable on macOS. See https://bugs.python.org/issue33725#msg343838
# In theory, using "spawn" start_method would work, but that triggers a bug in pandarallel
# (https://github.com/nalepae/pandarallel/issues/72).
USE_MULTIPROCESSING = FORCE_MULTIPROCESSING or platform.system() != "Darwin"
if not USE_MULTIPROCESSING:
    _log.info(
        "Parallel code via multiprocessing disabled on macOS. Set FORCE_MULTIPROCESSING env var to override."
    )

T = TypeVar("T")
R = TypeVar("R")
SeriesOrDataFrame = TypeVar("SeriesOrDataFrame", pd.Series, pd.DataFrame)


def parallel_map(func: Callable[[T], R], iterable: Iterable[T]) -> Iterable[R]:
    """Runs func on each item in iterable, in parallel if possible."""
    if USE_MULTIPROCESSING:
        # Our current GitHub Actions runner has 40 cores, so default to this
        # and fall back to a lower amount as necessary.
        processes = min(os.cpu_count(), 40)
        with get_context("spawn").Pool(processes=processes) as pool:
            # Always return an iterator to make sure the return type is consistent.
            return iter(pool.map(func, iterable))
    else:
        return map(func, iterable)


def pandas_parallel_apply(
    func: Callable[[T], R], series_or_dataframe: SeriesOrDataFrame
) -> SeriesOrDataFrame:
    """Calls parallel_apply() (from pandarallel) if safe, else just apply()."""
    if USE_MULTIPROCESSING:
        return series_or_dataframe.parallel_apply(func)
    else:
        return series_or_dataframe.apply(func)
