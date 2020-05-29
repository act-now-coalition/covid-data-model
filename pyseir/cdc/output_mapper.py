import itertools
import os
import us
import scipy
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, date
from multiprocessing import Pool
from string import Template
from pyseir import OUTPUT_DIR, load_data
from pyseir.utils import REF_DATE
from pyseir.cdc.parameters import (TEAM, MODEL, TARGETS, TARGETS_TO_NAMES,
                                   FORECAST_TIME_UNITS, FORECAST_WEEKS_NUM,
                                   QUANTILES, FORECAST_DATE, NEXT_EPI_WEEK,
                                   COLUMNS, DATE_FORMAT, Target,
                                   ForecastTimeUnit, ForecastTimeUnit,
                                   ForecastUncertainty)
from pyseir.cdc.utils import (target_column_name,
                              aggregate_timeseries,
                              smooth_timeseries,
                              number_of_time_units,
                              load_and_aggregate_observations)
from pyseir.inference.fit_results import load_inference_result, load_mle_model




"""
This mapper maps current pyseir model output to match cdc format.

Output file should have columns:
- forecast_date: the date on which the submitted forecast data was made available 
                 in YYYY-MM-DD format
- target: Values in the target column must be a character (string) and have format 
         "<day_num> day ahead <target_measure>" where day_num is number of 
         days since forecast_date to each date in forecast time range. 
- target_end_date: end date of forecast in YYYY-MM-DD format.
- location: 2 digit FIPS code
- type: "quantile"
- quantile: quantiles of forecast target measure, with format 0.###.
- value: value of target measure at given quantile and forecast date for given 
         location
- location_name: name of the location that can be useful to identify the 
                 location. 

For details on formatting, check:
https://github.com/reichlab/covid19-forecast-hub/blob/master/data-processed/README.md

The output includes:
- <forecast_date>-<team>-<model>_<fips>.csv
  File that contain the output with above columns for a specific fips.
- <forecast_date>-<team>-<model>.csv
  File that contain the output with above columns for all US states fips.
- metadata-CovidActNow.txt
  Metadata with most up-to-date forecast date.
  
Where default value of forecast_date, team and model can be found from 
corresponding global variables.
"""

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
REPORT_FOLDER = os.path.join(DIR_PATH, 'report')


class OutputMapper:
    """
    This mapper maps CAN SEIR model inference results to the format required
    for CDC model submission. For the given State FIPS code, it reads in the
    most up-to-date MLE inference (mle model + fit_results json file),
    and aligns the forecast with the latest observations.

    Quantile of forecast is derived assuming uncertainty from two
    sources: distribution of forecast error and distribution of forecast
    itself. Distribution of forecast error is obtained by collecting
    historical 1-day step forecast: y_t+1 - y^hat_t+1 | y_t with t varied from
    first day of observation to one day before the latest observation.

    For nth day forecast,
        forecast error ~ N(0, error_std)
        where error_std
        = (forecast + variance of historical 1-day error * s(n)) ** 0.5
    and s(n) is the factor that increase the forecast uncertainty based on
    n, currently s(n) is either 1 or n ** 0.5.

    The output has the columns required for CDC model ensemble
    submission (check description of results). It currently supports daily
    and weekly forecast.


    Attributes
    ----------
    model: SEIRModel
        SEIR model with MLE parameter estimates.
    fit_results: dict
        Maximum likelihood parameters inferred by fitting SEIR model to
        observed cases/deaths/hospitalizations.
    forecast_given_time_range: callable
        Makes forecast of a target during the forecast time window given a
        model's prediction of target and model's t_list (t steps since the
        inferred starting time of the epidemic, t0).
    targets: list(Target)
        List of Target objects.
    forecast_time_units: list(ForecastTimeUnit)
        List of ForecastTimeUnit objects, determines whether forecast target
        is aggregated by day or week. Currently the mapper only supports
        daily forecasts.
    forecast_uncertainty: ForecastUncertainty
        Determines how forecast uncertainty is adjusted based on number of
        days the forecast is made ahead and total days of observations.
        Should be interpretable by ForecastUncertainty. Currently supports:
        - ForecastUncertainty.DEFAULT: no adjustment
        - ForecastUncertainty.NAIVE: rescale the standard deviation by factor
          (days_ahead  ** 0.5)
    observations: dict(dict)
        Contains observed cumulative deaths, incident deaths,
        and incident hospitalizations, with target name as primary key and
        forecast time unit as secondary key, and corresponding time series of
        observations as values:
        <target>:
            <forecast time unit>: pd.Series
                With date string as index and observations as values.
        Observations for hospitalizations can be None if no
        cumulative hospitalization data is available for the FIPS code.
    errors: dict
        Contains all historical 1-day step absolute forecast error for each
        type of observation: cumulative death, incident death, incident
        hospitalization.
    result: pd.DataFrame
        Output that meets requirement for CDC model ensemble submission for
        given FIPS code.
        Contains columns:
        - forecast_date: datetime.datetime
          the date on which the submitted forecast data was made available in
          YYYY-MM-DD format
        - target: str
          Name of the forecast target, with format "<day_num> <unit>
          ahead <target_measure>" where day_num is number of days/weeks since
          forecast_date to each date in forecast time range, and unit is the
          time unit of the forecast, i.e. day or wk.
        - target_end_date: datetime.datetime
          last date of forecast in YYYY-MM-DD format.
        - location: str
          2 digit FIPS code.
        - type: str
          "quantile"
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
    forecast_time_units: list(str)
        Names of the time unit of forecast, should be interpretable by
        ForecastTimeUnit.
    forecast_date: datetime.date
        Date when the forecast is done, default the same day when the mapper
        runs.
    next_epi_week: epiweeks.week
        The coming epi weeks, with the start date of which the forecast time
        window begins.
    quantiles: list(float)
        Values between 0-1, which are the quantiles of the forecast target
        to collect. For default value check QUANTILES.
    forecast_uncertainty: str
        Determines how forecast uncertainty is adjusted based on number of
        days the forecast is made ahead and total days of observations.
        Should be interpretable by ForecastUncertainty. Currently supports:
        - 'default': no adjustment
        - 'naive': rescale the standard deviation by factor (days_ahead
                   ** 0.5)
    """
    def __init__(self,
                 fips,
                 targets=TARGETS,
                 forecast_time_units=FORECAST_TIME_UNITS,
                 forecast_date=FORECAST_DATE,
                 next_epi_week=NEXT_EPI_WEEK,
                 quantiles=QUANTILES,
                 forecast_uncertainty='naive'):

        self.fips = fips
        self.targets = [Target(t) for t in targets]
        self.forecast_time_units = [ForecastTimeUnit(u) for u in forecast_time_units]
        self.forecast_date = forecast_date
        self.forecast_time_range = [(datetime.fromordinal(next_epi_week.startdate().toordinal()) + timedelta(n))
                                    for n in range(FORECAST_WEEKS_NUM * 7)]
        self.quantiles = quantiles
        self.forecast_uncertainty = ForecastUncertainty(forecast_uncertainty)
        self.observations = load_and_aggregate_observations(fips=self.fips,
                                                            units=self.forecast_time_units,
                                                            targets=self.targets,
                                                            end_date=self.forecast_date)
        self.model = load_mle_model(self.fips)

        self.fit_results = load_inference_result(self.fips)

        self.errors = None
        self.result = None


    def generate_forecast(self, start_date=None, align_with_observations=True):
        """
        Generates a forecast ensemble given the model ensemble.

        Parameters
        ----------
        start_date: datetime.datetime
            First date of forecast, default to date of latest observation.
        align_with_observations: bool
            Whether shift the forecast to match the latest observation.

        Returns
        -------
        forecast: dict(pd.DataFrame)
            Contains forecast of target within the forecast time window.
            With targets as primary keys and units as secondary keys and
            corresponding forecast time series as values:
            <target>:
                <unit>: pd.DataFrame
                 With columns:
                    - target: forecat target name with format
                              '<n> <unit> ahead <target type>'
                              where n is the number of time units, unit can
                              be day or wk, and target type is 'cum death',
                              'inc death' or 'inc hosp'.
                    - value: str, maximum likelihood forecast value
                    - target_end_date: np.datetime64, date of the forecast
                    - forecast_days: days forward of forecast since lastest
                                     observation
        """
        # start date as date of latest observation
        start_date = start_date or datetime.strptime(self.observations['cum death']['day'].index[-1],
                                                     DATE_FORMAT)

        # record forecast 10 days before the last observation to enable
        # calculation of week ahead forecast
        forecast_days_since_ref_date = list(range((start_date - REF_DATE).days - 10,
                                               (self.forecast_time_range[-1] - REF_DATE).days + 1))
        forecast_dates = [timedelta(t) + REF_DATE for t in forecast_days_since_ref_date]

        forecast_given_time_range = \
            lambda forecast: np.interp(forecast_days_since_ref_date,
                                       [self.fit_results['t0'] + t for t in self.model.t_list],
                                       forecast)

        forecast = defaultdict(dict)
        for target in self.targets:
            for unit in self.forecast_time_units:
                if target is Target.INC_DEATH:
                    predictions = forecast_given_time_range(self.model.results['total_deaths_per_day'])

                elif target is Target.INC_HOSP:
                    predictions = forecast_given_time_range(
                        np.append([0], np.diff(self.model.results['HGen_cumulative']
                                             + self.model.results['HICU_cumulative'])))

                elif target is Target.CUM_DEATH:
                    predictions = forecast_given_time_range(self.model.results['D'])

                else:
                    raise ValueError(f'Target {target} is not implemented')

                dates, predictions = aggregate_timeseries(forecast_dates,
                                                            predictions,
                                                            unit,
                                                            target)
                df = pd.DataFrame(
                        {'value': predictions.clip(min=0),
                         'target_end_date': dates,
                         'forecast_days': [(t - start_date).days for t in dates]},
                        index=pd.DatetimeIndex(dates).strftime(DATE_FORMAT))

                n_units = number_of_time_units(self.forecast_date, dates, unit)
                df['target'] = list(target_column_name(n_units, target, unit))
                df['target_end_date'] = df['target_end_date'].astype('datetime64[D]')

                forecast[target.value][unit.value] = df

        if align_with_observations:
            forecast = self.align_forecast_with_observations(forecast)

        return forecast


    def align_forecast_with_observations(self, forecast):
        """
        Shift forecast curve so that its value on the date of lastest
        observation matches the observation.

        Parameters
        ----------
        forecast: dict(pd.DataFrame)
            Contains forecast of target within the forecast time window.
            With targets as primary keys and units as secondary keys and
            corresponding forecast time series as values:
            <target>:
                <unit>: pd.DataFrame
                 With columns:
                    - target: forecat target name with format
                              '<n> <unit> ahead <target type>'
                              where n is the number of time units, unit can
                              be day or wk, and target type is 'cum death',
                              'inc death' or 'inc hosp'.
                    - value: str, maximum likelihood forecast value
                    - target_end_date: np.datetime64, date of the forecast
                    - forecast_days: days forward of forecast since lastest
                                     observation

        Returns
        -------
        shifted_forecast: dict(pd.DataFrame)
            Contains forecast of target within the forecast time window
            shifted based on latest observations.
            With targets as primary keys and units as secondary keys and
            corresponding forecast time series as values:
            <target>:
                <unit>: pd.DataFrame
                 With columns:
                    - target: forecat target name with format
                              '<n> <unit> ahead <target type>'
                              where n is the number of time units, unit can
                              be day or wk, and target type is 'cum death',
                              'inc death' or 'inc hosp'.
                    - value: str, maximum likelihood forecast value
                    - target_end_date: np.datetime64, date of the forecast
                    - forecast_days: days forward of forecast since lastest
                                     observation

        """
        shifted_forecast = forecast.copy()
        # align with observations
        for target in self.targets:
            for unit in self.forecast_time_units:
                if self.observations[target.value][unit.value] is not None:
                    ref_observation_date = self.observations[target.value][unit.value].index[-1]
                    observation = self.observations[target.value][unit.value]

                    shifted_forecast[target.value][unit.value]['value'] += \
                        (observation.loc[ref_observation_date]
                       - forecast[target.value][unit.value]['value'].loc[ref_observation_date])

                    shifted_forecast[target.value][unit.value]['value'] = \
                        shifted_forecast[target.value][unit.value]['value'].clip(lower=0)

        return shifted_forecast


    def calculate_errors(self, unit=ForecastTimeUnit.DAY):
        """
        Collect distribution of historical absolute errors 1-day forecast abs(
        y_t - y^hat_t|y_t-1) for t from second earliest available observation to
        latest available observation.

        Parameters
        ----------
        unit: ForecastTimeUnit
            Time unit of forecast to calculate the error, default
            ForecastTimeUnit.DAY.

        Returns
        -------
          :  dict
            Contains all historical 1-day step absoluate forecast error for
            each type of observation: cumulative death, incident death,
            incident hospitalization.
        """
        self.errors = defaultdict(list)
        for date in self.observations['cum death'][unit.value].index[:-1]:
            forecast = self.generate_forecast(start_date=datetime.strptime(date, DATE_FORMAT))
            for target in self.targets:
                if self.observations[target.value][unit.value] is not None:
                    next_day = datetime.strftime(datetime.strptime(date, DATE_FORMAT) + timedelta(1),
                                                 DATE_FORMAT)
                    if next_day in self.observations[target.value][unit.value].index:
                        pred = forecast[target.value][unit.value]['value'].loc[next_day]
                        observ = self.observations[target.value][unit.value].loc[next_day]
                        self.errors[target.value].append(abs(pred - observ))

        self.errors = dict(self.errors)

        return self.errors


    def _calculate_forecast_quantiles(self, forecast, error_std, h, quantiles, baseline=0):
        """
        Generate forecast quantiles based on standard deviation of historical
        forecast errors, forecast, number of days forward. Values for
        given quantile are clipped at given baseline value.

        Currently supports two approaches:
        - default: no adjustment
        - naive: rescale the standard deviation by factor (days_ahead  **
                 0.5)

        Parameters
        ----------
        forecast: float
            Value of forecast.
        error_std: float
            Standard deviation of historical forecast errors.
        h: int or float
            Time step of forecast in days.
        quantiles: list or np.array
            Quantile
        baseline: float
            Minimum value to clip the values at each quantile.

        Returns
        -------
          :  np.array
            Data after the adjustment.
        """

        if self.forecast_uncertainty is ForecastUncertainty.DEFAULT:
            scale_factor = 1
        elif self.forecast_uncertainty is ForecastUncertainty.NAIVE:
            scale_factor = h**0.5
        else:
            raise ValueError(f'forecast accuracy adjustment {self.forecast_uncertainty} is not implemented')

        values = scipy.stats.norm(loc=forecast,
                                  scale=np.sqrt((error_std * scale_factor) ** 2 + forecast)) \
                      .ppf(quantiles)
        values = values.clip(min=baseline)

        return values

    def generate_forecast_quantiles(self, forecast, quantiles=QUANTILES):
        """
        Runs forecast of a target with given model.

        Parameters
        ----------
        forecast: dict
            Contains forecast of target within the forecast time window.
            With targets as primary keys and units as secondary keys and
            corresponding forecast time series as values:
            <target>:
                <unit>: pd.DataFrame
                 With columns:
                    - target: forecat target name with format
                              '<n> <unit> ahead <target type>'
                              where n is the number of time units, unit can
                              be day or wk, and target type is 'cum death',
                              'inc death' or 'inc hosp'.
                    - value: str, maximum likelihood forecast value
                    - target_end_date: np.datetime64, date of the forecast
                    - forecast_days: days forward of forecast since lastest
                                     observation

        quantiles: np.array or list
            Quantile of forecast.

        Returns
        -------
        forecast_quantiles: dict
            Contains forecast of target within the forecast time window.
            With targets as primary keys and units as secondary keys and
            corresponding forecast time series as values:
            <target>:
                <unit>: pd.DataFrame
                 With columns:
                    - target: forecat target name with format
                              '<n> <unit> ahead <target type>'
                              where n is the number of time units, unit can
                              be day or wk, and target type is 'cum death',
                              'inc death' or 'inc hosp'.
                    - value: str, maximum likelihood forecast value
                    - target_end_date: np.datetime64, date of the forecast
                    - quantile: days forward of forecast since lastest
                                     observation
        """
        forecast_quantiles = defaultdict(dict)
        for target_name in forecast:
            if target_name in self.errors:
                error_std = np.std(self.errors[target_name])
            else:
                error_std = np.sqrt(forecast[target_name]['day']['value'].iloc[0])

            # For cumulative death forecast should not fall below the lastest
            #  observed death.
            if Target(target_name) is Target.CUM_DEATH:
                baseline = \
                    self.observations[Target(target_name).value]['day'].iloc[-1]
            # For incident death/hospitalization, forecast should not fall
            # below 0
            else:
                baseline = 0

            for unit_name in forecast[target_name]:
                df = forecast[target_name][unit_name]\
                    .set_index(['target', 'target_end_date'])\
                    .apply(
                    lambda r: self._calculate_forecast_quantiles(
                        forecast=r.value,
                        error_std=error_std,
                        h=r.forecast_days,
                        quantiles=quantiles,
                        baseline=baseline),
                    axis=1).rename('value')

                df = df.explode().reset_index()
                df['quantile'] = np.tile(self.quantiles,
                                         forecast[target_name][unit_name].shape[0])
                df['type'] = 'quantile'
                forecast_quantiles[target_name][unit_name] = df

        return forecast_quantiles


    def run(self):
        """
        Makes MLE forecast, calculating historical forecast errors and
        quantiles of forecast.
        Results contain quantiles of the forecast targets and saves results
        to csv file.

        Returns
        -------
          :  pd.DataFrame
          Output that meets requirement for CDC model ensemble submission for
          given FIPS code. Contains columns:
            - forecast_date: datetime.datetime
              the date on which the submitted forecast data was made available in
              YYYY-MM-DD format
            - target: str
              Name of the forecast target, with format "<day_num> <unit> ahead
              <target_measure>" where day_num is number of days/weeks since
              forecast_date to each date in forecast time range, and unit is
              the time unit of the forecast, i.e. day or wk.
            - target_end_date: datetime.datetime
              Date of forecast target in YYYY-MM-DD format.
            - location: str
              2 digit FIPS code.
            - type: str
              "quantile"
            - quantile: str
              quantiles of forecast target measure, with format 0.###.
            - value: float
              Value of target measure at given quantile and forecast date for
              given location.
            - location_name: str
              Name of the state.
        """
        forecast = self.generate_forecast()
        self.calculate_errors()
        forecast_quantile = self.generate_forecast_quantiles(forecast)

        result = list()
        for target_name in forecast_quantile:
            for unit in self.forecast_time_units:
                result.append(forecast_quantile[target_name][unit.value])

        result = pd.concat(result)
        result = result[result['target_end_date'] >= self.forecast_date]

        result['location'] = str(self.fips)
        result['location_name'] = us.states.lookup(self.fips).name
        result['forecast_date'] = self.forecast_date
        result['forecast_date'] = result['forecast_date'].dt.strftime('%Y-%m-%d')
        result['quantile'] = result['quantile'].apply(lambda v: '%.3f' % v)
        result = result[~result['target'].apply(lambda s: 'wk ahead inc hosp' in s)]

        self.result = result[['forecast_date',
                              'location',
                              'location_name',
                              'target',
                              'target_end_date',
                              'type',
                              'quantile',
                              'value']]

        forecast_date = self.forecast_date.strftime(DATE_FORMAT)
        self.result.to_csv(os.path.join(REPORT_FOLDER,
                                        f'{forecast_date}_{TEAM}_{MODEL}_{self.fips}.csv'),
                           index=False)

        return self.result


    @classmethod
    def run_for_fips(cls, fips, kwargs=None):
        """
        Run OutputMapper for given State FIPS code.

        Parameters
        ----------
        fips: str
            State FIPS code

        Returns
        -------
        results: pd.DataFrame
            Output that meets requirement for CDC model ensemble submission for
            given FIPS code. For details on columns, refer description of
            self.results.
        """
        kwargs = kwargs or {}
        om = cls(fips, **kwargs)
        result = om.run()
        return result


    @classmethod
    def generate_metadata(cls):
        """
        Generates metadata file based on the template.
        """
        om = cls(fips='06')
        with open(os.path.join(DIR_PATH, 'metadata-CovidActNow_template.txt'),  'r') as f:
            metadata = f.read()

        combined_target_names = list(itertools.product([u.value for u in om.forecast_time_units],
                                     [t.value for t in om.targets]))
        names = [' ahead '.join(tup) for tup in combined_target_names]
        names = [v for v in names if 'wk ahead inc hosp' not in v]

        metadata = \
            Template(metadata).substitute(
            dict(Model_targets=', '.join(names),
                 forecast_startdate=om.forecast_time_range[0].strftime('%Y-%m-%d'),
                 Model_target_names=', '.join([TARGETS_TO_NAMES[t.value] for t in om.targets]),
                 model_name=MODEL,
                 team_name=TEAM)
        )

        output_f = open(os.path.join(REPORT_FOLDER, f'metadata-{TEAM}-{MODEL}.txt'), 'w')
        output_f.write(metadata)
        output_f.close()


def run_all(parallel=False, mapper_kwargs=None):
    """
    Prepares inference results for all whitelist States for CDC model
    ensemble submission.

    Parameters
    ----------
    parallel: bool
        Whether to run multiprocessing.
    mapper_kwargs: dict
        Contains parameters and values to override the default output mapper
        parameters (given in parameters.py).
    """
    mapper_kwargs = mapper_kwargs or {}
    if not os.path.exists(REPORT_FOLDER):
        os.mkdir(REPORT_FOLDER)

    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist['inference_ok'] == True]
    fips_list = list(df_whitelist['fips'].str[:2].unique())

    if parallel:
        p = Pool()
        results = p.map(OutputMapper.run_for_fips, fips_list)
        p.close()
    else:
        results = list()
        for fips in fips_list:
            result = OutputMapper.run_for_fips(fips, mapper_kwargs)
            results.append(result)

    forecast_date = FORECAST_DATE.strftime(DATE_FORMAT)

    results = pd.concat(results)
    results = results[COLUMNS].sort_values(COLUMNS)
    results.to_csv(os.path.join(REPORT_FOLDER, f'{forecast_date}-{TEAM}-{MODEL}.csv'),
                   index=False)

    OutputMapper.generate_metadata()
