import multiprocessing
import os
import platform
from typing import Callable, TypeVar, Iterable, List

from pandarallel import pandarallel
import pandas as pd


VISIBIBLE_PROGRESS_BAR = os.environ.get("PYSEIR_VERBOSITY") == "True"
pandarallel.initialize(progress_bar=VISIBIBLE_PROGRESS_BAR)

# multiprocessing is unreliable on macOS. See https://bugs.python.org/issue33725#msg343838
# In theory, using "spawn" start_method would work, but that triggers a bug in pandarallel
# (https://github.com/nalepae/pandarallel/issues/72).
USE_MULTIPROCESSING = platform.system() != "Darwin"

T = TypeVar("T")
R = TypeVar("R")
SERIES_OR_DF = TypeVar("SERIES_OR_DF", pd.Series, pd.DataFrame)


def parallel_map(func: Callable[[T], R], iterable: Iterable[T]) -> List[R]:
    """Runs func on each item in iterable, in parallel if possible."""
    if USE_MULTIPROCESSING:
        # Setting maxtasksperchild to one ensures that we minimize memory usage over time by creating
        # a new child for every task. Addresses OOMs we saw on highly parallel build machine.
        with multiprocessing.Pool(maxtasksperchild=1) as pool:
            return pool.map(func, iterable)
    else:
        return list(map(func, iterable))


def pandas_parallel_apply(
    func: Callable[[T], R], series_or_dataframe: SERIES_OR_DF
) -> SERIES_OR_DF:
    """Calls parallel_apply() (from pandarallel) if safe, else just apply()."""
    if USE_MULTIPROCESSING:
        return series_or_dataframe.parallel_apply(func)
    else:
        return series_or_dataframe.apply(func)
