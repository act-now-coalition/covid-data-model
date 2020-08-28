import structlog
from dataclasses import dataclass
from typing import Callable
from typing import List

import pandas as pd

# from libs import series_utils

rt_log = structlog.get_logger()


@dataclass
class InclusionCriteria:
    """Class for keeping track of inclusion criteria"""

    name: str
    evaluate: Callable[[pd.Series], bool]

    def __str__(self):
        return self.name


def get_full_test_suite() -> List[InclusionCriteria]:
    return [
        InclusionCriteria("Sufficient_Cumulative_Cases", lambda x: check_total_counts(x)),
        InclusionCriteria("Sufficient_Incident_Cases", lambda x: check_min_incident_counts(x)),
        InclusionCriteria("Sufficient_Number_of_Datapoints", lambda x: check_min_length(x)),
        # Additional Criteria to be included after confirming refactor is stable
        # InclusionCriteria("Sufficient_Recent_Data", lambda x: series_utils.has_recent_data(x)),
        # InclusionCriteria("Sufficient_Recent_Cases", lambda x: check_recent_cases(x)),
    ]


def evaluate_tests(s: pd.Series, tests: List[InclusionCriteria], log=structlog.getLogger()) -> bool:
    """

    Parameters
    ----------
    s: pd.Series

    tests: List[InclusionCriteria]
        A list of InclusionCriteria instances that will be independently evaluated for inclusion.
        The function short circuits on the first failure.
    log:
        Log instance with context about the region being tested.

    Returns
    -------
    True if all InclusionCriteria are satisfied, else False
    """
    for test in tests:
        if not test.evaluate(s):
            log.info("Region Excluded from Infection Rate Calc: ", test_failed=test.name)
            return False
    else:
        log.info("Region Passed all Infection Rate Inclusion Criteria")
        return True


def check_total_counts(s: pd.Series, threshold: int = 20) -> bool:
    return s.sum() > threshold


def check_min_incident_counts(s: pd.Series, threshold: int = 5) -> bool:
    return s.max() > threshold


def check_min_length(s: pd.Series, threshold: int = 20) -> bool:
    return len(s) > threshold


def check_recent_cases(s: pd.Series, threshold: int = 20, n_days_lookback: int = 14) -> bool:
    return s[-n_days_lookback:].sum() > threshold
