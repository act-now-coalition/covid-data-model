import json
import pickle
from enum import Enum
import numpy as np
import pandas as pd
from pyseir.utils import get_run_artifact_path, RunArtifact
from pyseir.inference.fit_results import load_inference_result, load_mle_model
from pyseir.ensembles.ensemble_runner import EnsembleRunner


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

TARGETS = ['cum death']
FORECAST_TIME_UNITS = ['day', 'wk']
QUANTILES = np.concatenate([[0.01, 0.025],
                            np.arange(0.05, 0.95, 0.05),
                            [0.975, 0.99]])

FORECAST_TIME_LIMITS = {'day': 130, 'wk': 20}


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
                 targets=TARGETS,
                 forecast_time_units=FORECAST_TIME_UNITS,
                 quantiles=QUANTILES,
                 type='quantile',
                 ensemble=True):
        self.fips = fips
        self.targets = targets
        self.forecast_time_units = forecast_time_units
        self.quantiles = quantiles
        self.type = type
        self.model = load_mle_model(self.fips)
        self.fit_results = load_inference_result(self.fips)
        self.ensemble = ensemble


    def load_inference_results(self):
        model = load_mle_model
        fit_results = load_inference_result
        return model, fit_results


    def calculate_posteriors(self):
        """

        """
        return

    def forecast(self, target):
        return

    def generate_metadata(self):
        return


