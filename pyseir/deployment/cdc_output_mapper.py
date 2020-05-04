import json
import pickle
from enum import Enum
import pandas as pd
from pyseir.utils import get_run_artifact_path, RunArtifact
from pyseir.load_data import load_inference_results


"""
This class maps current pyseir model output to match cdc format.

output file should have columns:
- forecast_date
- target
  
- target_end_date
- location
- type
- quantile
- value
"""

FORECAST_TIME_LIMITS = {'day': 130,
                        'wk': 20}

class Target(Enum):
    CUM_DEATH = 'cum death'
    INC_DEATH = 'inc death'
    CUM_HOSP = 'cum hosp'
    INC_HOSP = 'inc hosp'


class ForecastTimeUnit(Enum):
    DAY = 'day'
    WK = 'wk'


def target_column_name(n, target, time_unit):
    """

    """

    return f'{n} {time_unit.value} ahead {target.value}'



class CDCOutputMapper:
    def __init__(self,
                 fips,
                 targets,
                 forecast_time_units,
                 ):
        self.fips = fips
        self.targets = targets
        self.forecast_time_units = forecast_time_units
        self.model, self.fit_results = load_inference_results(self.fips)

    def forecast(self, target):

        return

    def generate_metadata(self):

        return


