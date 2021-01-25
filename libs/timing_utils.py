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
