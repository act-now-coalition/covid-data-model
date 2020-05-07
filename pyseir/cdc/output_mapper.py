import json
import pickle
from datetime import date, datetime, timedelta
import numpy as np
from enum import Enum
from epiweeks import Week, Year
from collections import defaultdict
from pyseir.utils import REF_DATE
import numpy as np
import pandas as pd
import inspect
from pyseir.cdc.utils import Target, ForecastTimeUnit, ForecastAccuracy, target_column_name
from pyseir.models.seir_model import SEIRModel
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

TARGETS = ['cum death', 'cum hosp', 'inc death', 'inc hosp']
FORECAST_TIME_UNITS = ['day']
QUANTILES = np.concatenate([[0.01, 0.025], np.arange(0.05, 0.95, 0.05), [0.975, 0.99]])

FORECAST_TIME_LIMITS = {'day': 130, 'wk': 20}

FORECAST_EPI_WEEK = Week(Year.thisyear().year, Week.thisweek().week + 1)

FORECAST_DATE = datetime.today()

class OutputMapper:
    """
    Currently supports daily forecast.
    """
    def __init__(self,
                 fips,
                 N_samples=5000,
                 targets=TARGETS,
                 forecast_date=FORECAST_DATE,
                 forecast_epi_week=FORECAST_EPI_WEEK,
                 forecast_time_units=FORECAST_TIME_UNITS,
                 quantiles=QUANTILES,
                 forecast_accuracy='no_adjust'):

        self.fips = fips
        self.N_samples = N_samples
        self.targets = [Target(t) for t in targets]
        self.forecast_date=forecast_date
        self.forecast_end_date=forecast_epi_week.enddate()
        self.forecast_time_range = [datetime.fromordinal(forecast_epi_week.startdate().toordinal()) + timedelta(n)
                                    for n in range(7)]
        self.forecast_time_units = [ForecastTimeUnit(u) for u in forecast_time_units]
        self.quantiles = quantiles
        self.type = type
        self.model = load_mle_model(self.fips)
        self.fit_results = load_inference_result(self.fips)

        forecast_days_since_ref_date = [(t - REF_DATE).days for t in self.forecast_time_range]
        self.forecast = lambda forecast, t_list: \
            np.interp(forecast_days_since_ref_date,
                      [self.fit_results['t0'] + t for t in t_list],
                      forecast)
        self.forecast_accuracy = ForecastAccuracy(forecast_accuracy)

    def run_model_ensemble(self, override_param_names=['R0', 'I_initial', 'E_initial', 'suppression_policy']):
        """

        """
        override_params = {k: v for k, v in self.model.__dict__.items() if k in override_param_names}
        er = EnsembleRunner(fips=self.fips)
        model_ensemble = er._model_ensemble(override_params=override_params,
                                            N_samples=self.N_samples)
        return model_ensemble

    def forecast_target(self, model, target, unit):
        """

        """
        if target is Target.INC_DEATH:
            target_forecast = self.forecast(model.results['total_deaths_per_day'], model.t_list)

        elif target is Target.INC_HOSP:
            target_forecast = self.forecast(np.append([0],
                                                      np.diff(model.results['HGen_cumulative']
                                                            + model.results['HICU_cumulative'])),
                                            model.t_list)

        elif target is Target.CUM_DEATH:
            target_forecast = self.forecast(model.results['D'], model.t_list)

        elif target is Target.CUM_HOSP:
            target_forecast = self.forecast(model.results['HGen_cumulative'] + model.results['HICU_cumulative'],
                                            model.t_list)

        else:
            raise ValueError(f'Target {target} is not implemented')

        target_forecast = pd.Series(target_forecast,
                                    index=target_column_name([(t - self.forecast_date).days for t in
                                                              self.forecast_time_range],
                                                              target, unit))

        return target_forecast

    def generate_forecast_ensemble(self, model_ensemble=None):
        """

        :param model_ensemble:
        :return:
        """
        if model_ensemble is None:
            model_ensemble = self.run_model_ensemble()

        forecast_ensemble = defaultdict(list)
        for target in self.targets:
            for unit in self.forecast_time_units:
                for model in model_ensemble:
                    target_forecast = self.forecast_target(model, target, unit).fillna(0)
                    target_forecast[target_forecast<0] = 0
                    forecast_ensemble[target.value].append(target_forecast)
            forecast_ensemble[target.value] = pd.concat(forecast_ensemble[target.value], axis=1)
        return forecast_ensemble

    def _adjust_forecast_dist(self, l, h, T):
        """

        """
        if self.forecast_accuracy is ForecastAccuracy.DEFAULT:
            return (l + (l - l.mean()) * (1 + h ** 0.5)).clip(0)
        elif self.forecast_accuracy is ForecastAccuracy.NO_ADJUST:
            return l
        else:
            raise ValueError(f'forecast accuracy adjustment {self.forecast_accuracy} is not implemented')


    def generate_quantile_output(self, forecast_ensemble=None):
        """

        """
        if forecast_ensemble is None:
            forecast_ensemble = self.generate_forecast_ensemble()

        quantile_output = list()
        for target_name in forecast_ensemble:
            target_output = \
                forecast_ensemble[target_name].apply(
                    lambda l: np.quantile(self._adjust_forecast_dist(l, int(l.name.split(' ')[0]),
                                                                     (self.forecast_date - REF_DATE).days),
                                          self.quantiles), axis=1).rename('value')
            target_output = target_output.explode().reset_index().rename(columns={'index': 'target'})
            target_output['quantile'] = np.tile(['%.3f' % q for q in self.quantiles],
                                                forecast_ensemble[target_name].shape[0])
            quantile_output.append(target_output)

        quantile_output = pd.concat(quantile_output, axis=0)
        quantile_output['location'] = self.fips
        quantile_output['target_end_date'] = self.forecast_end_date
        quantile_output['type'] = 'quantile'

        return quantile_output

    def generate_point_output(self):
        """

        """
        point_output = defaultdict(list)
        for target in self.targets:
            for unit in self.forecast_time_units:
                target_forecast = self.forecast_target(self.model, target, unit)
                point_output['value'].extend(target_forecast)
                point_output['target'].extend(target_column_name([(t - self.forecast_date).days for t in
                                                                  self.forecast_time_range],
                                                                  target, unit))

        point_output = pd.DataFrame(point_output)
        point_output['location'] = self.fips
        point_output['target_end_date'] = self.forecast_end_date
        point_output['type'] = 'point'

        return point_output

    def generate_metadata(self):

        return

    def run(self):
        quantile_output = self.generate_quantile_output()
        point_output = self.generate_point_output()
        output = pd.concat([quantile_output, point_output])
        metadata = self.generate_metadata()
        return output




