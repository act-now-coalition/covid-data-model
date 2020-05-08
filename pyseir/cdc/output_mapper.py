import itertools
import os
import us
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import date, datetime, timedelta
from enum import Enum
from multiprocessing import Pool
from epiweeks import Week, Year
from string import Template
from pyseir import OUTPUT_DIR, load_data
from pyseir.utils import REF_DATE
from pyseir.cdc.utils import Target, ForecastTimeUnit, ForecastUncertainty, target_column_name
from pyseir.inference.fit_results import load_inference_result, load_mle_model
from pyseir.ensembles.ensemble_runner import EnsembleRunner


"""
This mapper maps current pyseir model output to match cdc format.

Output file should have columns:
- forecast_date: the date on which the submitted forecast data was made available in YYYY-MM-DD format
- target: Values in the target column must be a character (string) and have format "<day_num> day ahead <target_measure>"
          where day_num is number of days since forecast_date to each date in forecast time range. 
- target_end_date: last date of forecast in YYYY-MM-DD format, will always be Saturday as defined by epi weeks.
- location: 2 digit FIPS code
- type: "quantile" or "point"
- quantile: quantiles of forecast target measure, with format 0.###.
- value: value of target measure at given quantile and forecast date for given location.
and optional: 
- location_name: name of the location that can be useful to identify the location. 

For details on formatting, check:
https://github.com/reichlab/covid19-forecast-hub/blob/master/data-processed/README.md

The output includes:
- <forecast_date>_<team>_<model>_<fips>.csv
  File that contain the output with above columns for a specific fips.
- <forecast_date>_<team>_<model>.csv
  File that contain the output with above columns for all US states fips.
- metadata-CovidActNow.txt
  Metadata with most up-to-date forecast date.
  
Where default value of forecast_date, team and model can be found from corresponding global variables.
"""


TEAM = 'CovidActNow'
MODEL = 'SEIR_CAN'

# type of target measures
TARGETS = ['cum death', 'inc death', 'inc hosp']

# names of target measures that will be used to generate metadata
TARGETS_TO_NAMES = {'cum death': 'cumulative deaths',
                    'inc death': 'incident deaths',
                    'inc hosp': 'incident hospitalizations'}

# units of forecast target, currently only supporting daily forecast.
FORECAST_TIME_UNITS = ['day']
# number of weeks ahead for forecast.
FORECAST_WEEKS_NUM = 2
# Default quantiles required by CDC.
QUANTILES = np.concatenate([[0.01, 0.025], np.arange(0.05, 0.95, 0.05), [0.975, 0.99]])
# Next epi week. Epi weeks starts from Sunday and ends on Saturday.
NEXT_EPI_WEEK = Week(Year.thisyear().year, Week.thisweek().week + 1)
# Time of forecast, default date when this runs.
FORECAST_DATE = datetime.today()

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUTPUT_FOLDER = os.path.join(OUTPUT_DIR, 'pyseir', 'cdc')


class OutputMapper:
    """
    This mapper maps CAN SEIR model inference results to the format required
    for CDC model submission. For the given State FIPS code, it reads in the
    most up-to-date MLE inference (mle model + fit_results json file), and runs
    the model ensemble when fixing the parameters varied for model fitting at
    their MLE estimate, aiming to obtain the uncertainty associated with
    the prior distribution of the parameters not varied during model fitting.
    This for sure will underestimate the level of uncertainty of the forecast
    since it does not take into account the parameters varied for MLE
    inference or the model's likelihood profile. However, neither is included
    for two reasons: 1) the likelihood function of the forecast is unknown;
    2) The error associated with the parameters for inference is generally
    small, so their variations are also relatively small.

    The output has the columns required for CDC model ensemble
    submission (check description of results). It currently supports daily
    forecast.

    Attributes
    ----------
    model: SEIRModel
        SEIR model with MLE parameter estimates.
    forecast_given_time_range: callable
        Makes forecast of a target during the forecast time window given a
        model's prediction of target and model's t_list (t steps since the
        inferred starting time of the epidemic, t0).
    targets: list(Target)
        List of Target objects.
    forecast_time_units: list(ForecastTimeUnit)
        List of ForecastTimeUnit objects, determines whether forecast target
        is aggregated by day or week. Currently the mapper only supports
        daily forecast.
    forecast_uncertainty: ForecastUncertainty
        Determines how forecast uncertainty is adjusted based on number of
        days the forecast is made ahead and total days of observations.
        Should be interpretable by ForecastUncertainty. Currently supports:
        - ForecastUncertainty.DEFAULT: no adjustment
        - ForecastUncertainty.NAIVE: rescale the standard deviation by factor (1
                                     + days_ahead  ** 0.5)
    result: pd.DataFrame
        Output that meets requirement for CDC model ensemble submission for
        given FIPS code.
        Contains columns:
        - forecast_date: datetime.datetime
          the date on which the submitted forecast data was made available in
          YYYY-MM-DD format
        - target: str
          Name of the forecast target, with format "<day_num> day ahead
          <target_measure>" where day_num is number of days since
          forecast_ate to each date in forecast time range.
        - target_end_date: datetime.datetime
          last date of forecast in YYYY-MM-DD format.
        - location: str
          2 digit FIPS code.
        - type: str
          "quantile" or "point"
        - quantile: str
          quantiles of forecast target measure, with format 0.###.
        - value: float
          Value of target measure at given quantile and forecast date for
          given location.
        - location_name: str
          Name of the state.


    Parameters
    ----------
    fips: str
        State or County FIPS code
    N_samples: int
        Number SEIR model parameter sets of sample.
    targets: list(str)
        Names of the targets to forecast, should be interpretable by Target.
    forecast_date: datetime.date
        Date when the forecast is done, default the same day when the mapper
        runs.
    next_epi_week: epiweeks.week
        The coming epi weeks, with the start date of which the forecast time
        window begins.
    forecast_time_units: list(str)
        Time units of the forecast target, should be interpretable by
        ForecastTimeUnit, currently the mapper only supports unit 'day'.
    quantiles: list(float)
        Values between 0-1, which are the quantiles of the forecast target
        to collect. For default value check QUANTILES.
    forecast_uncertainty: str
        Determines how forecast uncertainty is adjusted based on number of
        days the forecast is made ahead and total days of observations.
        Should be interpretable by ForecastUncertainty. Currently supports:
        - 'default': no adjustment
        - 'naive': rescale the standard deviation by factor (1 + days_ahead
                   ** 0.5)
    """
    def __init__(self,
                 fips,
                 N_samples=1000,
                 targets=TARGETS,
                 forecast_date=FORECAST_DATE,
                 next_epi_week=NEXT_EPI_WEEK,
                 forecast_time_units=FORECAST_TIME_UNITS,
                 quantiles=QUANTILES,
                 forecast_uncertainty='default'):

        self.fips = fips
        self.N_samples = N_samples
        self.targets = [Target(t) for t in targets]
        self.forecast_time_units = [ForecastTimeUnit(u) for u in forecast_time_units]
        self.forecast_date=forecast_date
        self.forecast_time_range = [datetime.fromordinal(next_epi_week.startdate().toordinal()) + timedelta(n)
                                    for n in range(FORECAST_WEEKS_NUM * 7)]

        self.quantiles = quantiles
        self.forecast_uncertainty = ForecastUncertainty(forecast_uncertainty)

        self.model = load_mle_model(self.fips)

        self.fit_results = load_inference_result(self.fips)
        forecast_days_since_ref_date = [(t - REF_DATE).days for t in self.forecast_time_range]
        self.forecast_given_time_range = \
            lambda forecast, t_list: np.interp(forecast_days_since_ref_date,
                                               [self.fit_results['t0'] + t for t in t_list],
                                               forecast)

        self.result = None

    def run_model_ensemble(self, override_param_names=['R0', 'I_initial', 'E_initial', 'suppression_policy']):
        """
        Get model ensemble by running models under different parameter sets
        sampled from parameter prior distributions.

        Parameters
        ----------
        override_param_names: list(str)
            Names of model parameters to override. Default list include the
            parameters varied for MLE inference.

        Returns
        -------
        model_ensemble: np.array(SEIRModel)
            SEIR models ran under parameter sets randomly generated
            from the parameter prior distributions.
        chi_squares: np.array(float)
            Chi squares when fitting each model in model_ensemble to
            to observed cases, deaths w/o hospitalizations.
        """

        override_params = {k: v for k, v in self.model.__dict__.items() if k in override_param_names}
        override_params.update({k: v for k, v in self.fit_results.items() if k in ['eps', 't_break', 'test_fraction']})
        er = EnsembleRunner(fips=self.fips)
        model_ensemble, chi_squares = er.model_ensemble(
            override_params=override_params, N_samples=self.N_samples, chi_square=True)

        return model_ensemble, chi_squares


    def forecast_target(self, model, target, unit):
        """
        Runs forecast of a target with given model.

        Parameters
        ----------
        model: SEIRModel
            SEIR model to run the forecast.
        target: Target
            The target to forecast.
        unit: ForecastTimeUnit
            Time unit to aggregate the forecast. Currently supports
            ForecastTimeUnit.DAY

        Returns
        -------
        target_forecast: np.array
            Forecast of target at given unit (currently only supports daily
            forecast), with shape (len(self.forecast_time_range),)
        """
        if unit is ForecastTimeUnit.DAY:
            if target is Target.INC_DEATH:
                target_forecast = self.forecast_given_time_range(model.results['total_deaths_per_day'], model.t_list)

            elif target is Target.INC_HOSP:
                target_forecast = self.forecast_given_time_range(np.append([0],
                                                                 np.diff(model.results['HGen_cumulative']
                                                                       + model.results['HICU_cumulative'])),
                                                                 model.t_list)

            elif target is Target.CUM_DEATH:
                target_forecast = self.forecast_given_time_range(model.results['D'], model.t_list)

            else:
                raise ValueError(f'Target {target} is not implemented')

            target_forecast = pd.Series(target_forecast,
                                        index=target_column_name([(t - self.forecast_date).days for t in
                                                                  self.forecast_time_range],
                                                                  target, unit))

            return target_forecast
        else:
            raise ValueError(f'Forecast time unit {unit} is not supported')


    def generate_forecast_ensemble(self, model_ensemble=None):
        """
        Generates forecast ensemble given the model ensemble.

        Parameters
        ----------
        model_ensemble: list(SEIRModel)
            List of SEIR models ran under parameter sets randomly generated
            from the parameter prior distributions.

        Returns
        -------
        forecast_ensemble: dict(pd.DataFrame)
            Contains forecast of target within the forecast time window run by
            each model from the model ensemble. With "<day_num> day ahead
            <target_measure>" as index, and corresponding value from each model
            as columns.
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

    def _adjust_forecast_dist(self, data, h, T):
        """
        Rescale forecast standard deviation by streching/shrinking the forecast
        distribution around the mean. Currently supports two approach:
        - default: no adjustment
        - naive: rescale the standard deviation by factor (1 + days_ahead  **
                 0.5)

        Parameters
        ----------
        data: np.array or list
            Data sample from the distribution.
        h: int or float
            Time step of forecast
        T: int or float
            Total days of projection before the forecast starts.

        Returns
        -------
          :  np.array
            Data after the adjustment.
        """
        data = np.array(data)
        if self.forecast_uncertainty is ForecastUncertainty.DEFAULT:
            return data
        elif self.forecast_uncertainty is ForecastUncertainty.NAIVE:
            return (data + (data - data.mean()) * (1 + h ** 0.5)).clip(0)
        else:
            raise ValueError(f'forecast accuracy adjustment {self.forecast_uncertainty} is not implemented')

    def _weighted_quantiles(self, quantile, data, weights):
        """
        Calculate quantile of data with given weights.

        Parameters
        ----------
        quantile: np.array of list
            Quantile to find corresponding data value.
        data: np.array or list
            Data sample
        weights: np.array
            Weight of each data point.

        Returns
        -------
          :  np.array
            Value of data at given quantile.
        """
        sorted_idx = np.argsort(data)
        cdf = weights[sorted_idx].cumsum() / weights[sorted_idx].cumsum().max()

        return np.interp(quantile, cdf, data[sorted_idx])

    def generate_quantile_result(self, forecast_ensemble, chi_squares):
        """
        Generates result that contains the quantiles of the forecast with
        format required for CDC model ensemble submission.

        Parameters
        ----------
        forecast_ensemble: dict
            Contains forecast of target within the forecast time window run by
            each model from the model ensemble. With "<day_num> day ahead
            <target_measure>" as index, and corresponding value from each model
            as columns.
        chi_squares: np.array(float)
            Chi squares obtains by fitting each model (which makes the
            forecast in forecast ensemble) to observed cases, deaths w/o
            hospitalizations.


        Returns
        -------
        quantile_result: pd.DataFrame
            Contains the quantiles of the forecast with format required for
            CDC model ensemble submission. For info on columns,
            check description of self.results.
        """

        quantile_result = list()
        for target_name in forecast_ensemble:
            target_output = \
                forecast_ensemble[target_name].apply(
                    lambda l: self._weighted_quantiles(self.quantiles,
                                                       self._adjust_forecast_dist(l, int(l.name.split(' ')[0]),
                                                                                 (self.forecast_date - REF_DATE).days),
                                                       1 / (1 + chi_squares - chi_squares.min())),
                    axis=1).rename('value')
            target_output = target_output.explode().reset_index().rename(columns={'index': 'target'})
            target_output['quantile'] = np.tile(['%.3f' % q for q in self.quantiles],
                                                forecast_ensemble[target_name].shape[0])
            quantile_result.append(target_output)

        quantile_result = pd.concat(quantile_result, axis=0)
        quantile_result['location'] = str(self.fips)
        quantile_result['location_name'] = us.states.lookup(self.fips).name
        quantile_result['target_end_date'] = self.forecast_time_range[-1]
        quantile_result['type'] = 'quantile'

        return quantile_result

    def generate_point_result(self):
        """
        Generates result that contains the point estimate of the forecast with
        format required for CDC model ensemble submission.

        Parameters
        ----------
        forecast_ensemble: dict
            Contains forecast of target within the forecast time window run by
            each model from the model ensemble. With "<day_num> day ahead
            <target_measure>" as index, and corresponding value from each model
            as columns.

        Returns
        -------
        point_result: pd.DataFrame
            Contains the MLE point estimate of forecast with format
            required for CDC model ensemble submission. For info on columns,
            check description of self.results.

        """
        point_result = defaultdict(list)
        for target in self.targets:
            for unit in self.forecast_time_units:
                target_forecast = self.forecast_target(self.model, target, unit)
                point_result['value'].extend(target_forecast)
                point_result['target'].extend(target_column_name([(t - self.forecast_date).days for t in
                                                                  self.forecast_time_range],
                                                                  target, unit))

        point_result= pd.DataFrame(point_result)
        point_result['location'] = str(self.fips)
        point_result['location_name'] = us.states.lookup(self.fips).name
        point_result['target_end_date'] = self.forecast_time_range[-1]
        point_result['type'] = 'point'

        return point_result

    def run(self):
        """
        Runs forecast ensemble, results that contain quantiles and point of
        the forecast targets and save results to csv file.

        Returns
        -------
          :  pd.DataFrame
          Output that meets requirement for CDC model ensemble submission for
          given FIPS code. Contains columns:
            - forecast_date: datetime.datetime
              the date on which the submitted forecast data was made available in
              YYYY-MM-DD format
            - target: str
              Name of the forecast target, with format "<day_num> day ahead
              <target_measure>" where day_num is number of days since
              forecast_ate to each date in forecast time range.
            - target_end_date: datetime.datetime
              last date of forecast in YYYY-MM-DD format.
            - location: str
              2 digit FIPS code.
            - type: str
              "quantile" or "point"
            - quantile: str
              quantiles of forecast target measure, with format 0.###.
            - value: float
              Value of target measure at given quantile and forecast date for
              given location.
            - location_name: str
              Name of the state.
        """
        models, chi_squares = self.run_model_ensemble()
        forecast_ensemble = self.generate_forecast_ensemble(models)
        quantile_result = self.generate_quantile_result(forecast_ensemble, chi_squares)
        point_result = self.generate_point_result()
        self.result = pd.concat([quantile_result, point_result])
        for col in ['location', 'quantile']:
            self.result[col] = self.result[col].apply('="{}"'.format)
        forecast_date = self.forecast_date.strftime('%Y-%m-%d')
        self.result.to_csv(os.path.join(OUTPUT_FOLDER,
                                        f'{forecast_date}_{TEAM}_{MODEL}_{self.fips}.csv'),
                           index=False)

        return self.result

    @classmethod
    def run_for_fips(cls, fips):
        """
        Run OutputMapper for given State FIPS code.
        """
        om = cls(fips)
        result = om.run()
        return result

    @classmethod
    def generate_metadata(cls):
        """
        Generates metadata file based on the template.
        """
        om = cls(fips='06')
        with open(os.path.join(DIR_PATH, 'metadata-CovidActNow_template.txt'), 'r') as f:
            metadata = f.read()
        f.close()

        combined_target_names = list(itertools.product([u.value for u in om.forecast_time_units],
                                     [t.value for t in om.targets]))
        metadata = \
            Template(metadata).substitute(
            dict(Model_targets=', '.join([' ahead '.join(tup) for tup in combined_target_names]),
                 forecast_startdate=om.forecast_time_range[0].strftime('%Y-%m-%d'),
                 Model_target_names=', '.join([TARGETS_TO_NAMES[t.value] for t in om.targets]),
                 model_name=MODEL,
                 team_name=TEAM)
        )

        with open(os.path.join(OUTPUT_FOLDER, 'metadata-CovidActNow.txt'), 'w') as output_f:
            output_f.write(metadata)
        output_f.close()


def run_all(parallel=False):
    """
    Prepares inference results for all whitelist States for CDC model
    ensemble submission.

    Parameters
    ----------
    parallel: bool
        Whether to run multiprocessing.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist['inference_ok'] == True]
    #fips_list = list(df_whitelist['fips'].str[:2].unique())
    fips_list = ['06']

    if parallel:
        p = Pool()
        results = p.map(OutputMapper.run_for_fips, fips_list)
        p.close()
    else:
        results = list()
        for fips in fips_list:
            result = OutputMapper.run_for_fips(fips)
            results.append(result)

    forecast_date = FORECAST_DATE.strftime('%Y-%m-%d')

    results = pd.concat(results)
    results.to_csv(os.path.join(OUTPUT_FOLDER, f'{forecast_date}_{TEAM}_{MODEL}.csv'),
                   index=False)

    OutputMapper.generate_metadata()
