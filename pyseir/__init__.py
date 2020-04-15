import pathlib
import os, math

import multiprocessing.pool

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output"


def divide_up_pool():
    """Divides up a large pool of cpus to be able to run parallelized runs, also in parallel.

    The first integer is the total available for a given stats/county.
    The latter integer is the number of states to run at the same time.

    Returns:
        (tuple[int, int]) -- a tuple of
    """
    total = os.cpu_count()
    if total < 20:
        return total, 1
    if total < 52:
        return math.floor(total/4), 4
    return math.floor(total/8), 8

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
