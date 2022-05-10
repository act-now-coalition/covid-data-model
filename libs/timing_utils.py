from typing import Optional
import time as time_stdlib
import structlog
import contextlib

_logger = structlog.get_logger()


@contextlib.contextmanager
def time(description: Optional[str] = None, **logging_args):

    start = time_stdlib.time()
    yield
    elapsed = time_stdlib.time() - start
    _logger.info(f"Elapsed: {elapsed:.1f}s", description=description, **logging_args)


from functools import wraps
from time import time as std_time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = std_time()
        result = f(*args, **kw)
        te = std_time()
        _logger.info(f"function: {f.__name__} executed in {te-ts} seconds.")
        return result

    return wrap
