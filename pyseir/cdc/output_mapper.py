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
                              load_state_level_death_data,
                              load_us_level_death_data,
                              aggregate_timeseries,
                              smooth_timeseries,
                              number_of_time_units,
                              load_and_aggregate_observations,
                              smooth_observations,
                              ppf_from_data,
                              random_sample_from_ppf)
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
REPORT_FOLDER = os.path.join(DIR_PATH, 'report',
                             datetime.strftime(FORECAST_DATE, DATE_FORMAT))


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

        self.raw_observations = load_and_aggregate_observations(fips=self.fips,
                                                            units=self.forecast_time_units,
                                                            targets=self.targets,
                                                            end_date=self.forecast_date)
        self.observations = smooth_observations(self.raw_observations)
        self.model = load_mle_model(self.fips)

        self.fit_results = load_inference_result(self.fips)

        self.raw_forecast = None
        self.errors = None
        self.result = None

    @staticmethod
    def forecast_format(forecast):
        """

        """
        for c in COLUMNS:
            if c not in forecast.columns:
                raise ValueError(f'column {c}' is missing)

        if forecast['quantile'].dtype is float:
            forecast['quantile'] = forecast['quantile'].apply(lambda v: '%.3f' % v)
        else forecast['quantile'].dtype is float:
            return



    def generate_raw_forecast(self):
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

        # start date as one day before the forecast date
        forecast_dates = [timedelta(self.fit_results['t0'] + t)
                          + REF_DATE for t in self.model.t_list]

        forecast = defaultdict(dict)
        for target in self.targets:
            for unit in self.forecast_time_units:
                if target is Target.INC_DEATH:
                    predictions = self.model.results['total_deaths_per_day']

                elif target is Target.INC_HOSP:
                    predictions = np.append([0], np.diff(self.model.results['HGen_cumulative']
                                                       + self.model.results['HICU_cumulative']))

                elif target is Target.CUM_DEATH:
                    predictions = self.model.results['D']

                else:
                    raise ValueError(f'Target {target} is not implemented')

                dates, predictions = aggregate_timeseries(forecast_dates,
                                                          predictions,
                                                          unit,
                                                          target)
                df = pd.DataFrame(
                        {'value': predictions.clip(min=0),
                         'target_end_date': dates,
                         'forecast_days': [(t.date() - self.forecast_date.date()).days for t in dates]},
                        index=pd.DatetimeIndex(dates).strftime(DATE_FORMAT))

                n_units = number_of_time_units(self.forecast_date, dates, unit)
                df['target'] = list(target_column_name(n_units, target, unit))
                df['target_end_date'] = df['target_end_date'].astype('datetime64[D]')
                df['type'] = 'point'

                forecast[target.value][unit.value] = df

        self.raw_forecast = dict(forecast)

        return self.raw_forecast


    def align_forecast_with_observations(self, ref_date=None, forecast=None,
                                         targets=None,
                                         units=None):
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
        forecast = forecast or self.raw_forecast
        units = units or self.forecast_time_units
        targets = targets or self.targets
        shifted_forecast = forecast.copy()
        # align with observations (not including weekly cumulative death
        # since it be generated based on daily cumulative death).
        for target in targets:
            for unit in units:
                if not ((target is Target.CUM_DEATH) & (unit is ForecastTimeUnit.WK)):
                    if self.observations[target.value][unit.value] is not None:
                        ref_observation_date = ref_date or self.observations[target.value][unit.value].index[-1]

                        observation = self.observations[target.value][unit.value]

                        shifted_forecast[target.value][unit.value]['value'] += \
                            (observation.loc[ref_observation_date]
                           - forecast[target.value][unit.value]['value'].loc[ref_observation_date])

                        shifted_forecast[target.value][unit.value]['value'] = \
                            shifted_forecast[target.value][unit.value]['value'].clip(lower=0)

        if ((Target.CUM_DEATH in targets) & (ForecastTimeUnit.WK in units)):
            shifted_forecast[Target.CUM_DEATH.value][ForecastTimeUnit.WK.value]['value'] = \
                shifted_forecast[Target.CUM_DEATH.value][ForecastTimeUnit.DAY.value]['value'].loc[
                    list(shifted_forecast[Target.CUM_DEATH.value][ForecastTimeUnit.WK.value].index)]


        return shifted_forecast


    def calculate_errors(self, window=30, unit=ForecastTimeUnit.DAY):
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
        window = window or 0

        for target in self.targets:
            if self.raw_observations[target.value][unit.value] is not None:
                observed_dates = list(self.raw_observations[target.value][unit.value].index)
                for n in range(len(observed_dates) - 1):
                    date = observed_dates[n]
                    next_date = observed_dates[n+1]
                    if date in self.raw_forecast[target.value][unit.value].index:
                        forecast = self.align_forecast_with_observations(ref_date=date,
                                                                         targets=[target],
                                                                         units=[ForecastTimeUnit.DAY])

                        pred = forecast[target.value][unit.value]['value'].loc[next_date]
                        observ = self.raw_observations[target.value][unit.value].loc[next_date]
                        self.errors[target.value].append(abs(observ - pred))

                self.errors[target.value] = self.errors[target.value][-window:]

        self.errors = dict(self.errors)

        return self.errors


    def _calculate_forecast_quantiles(self, forecast, error_std, h,
                                      quantiles, baseline=0):
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

        error_std = float(error_std)
        forecast = float(forecast)
        values = scipy.stats.norm(loc=forecast,
                                  scale=(error_std**2+forecast)**0.5*scale_factor)\
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
                    - target: forecast target name with format
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
                error_std = np.sqrt(forecast[target_name][
                    'day']['value'].iloc[0])

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
                    .set_index(['target', 'target_end_date'])

                df = df[df['forecast_days'] > 0]

                df = df.apply(
                    lambda r: self._calculate_forecast_quantiles(
                        forecast=r.value,
                        error_std=error_std,
                        h=r.forecast_days,
                        quantiles=quantiles,
                        baseline=baseline),
                    axis=1).rename('value')

                num_of_targets = df.shape[0]

                df = df.explode().reset_index()
                df['quantile'] = np.tile(self.quantiles, num_of_targets)
                df['type'] = 'quantile'
                df['quantile'] = df['quantile'].apply(lambda v: '%.3f' % v)
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
        self.raw_forecast = self.generate_raw_forecast()
        forecast = self.align_forecast_with_observations()
        self.calculate_errors()
        forecast_quantile = self.generate_forecast_quantiles(forecast)

        result = list()
        for target_name in forecast_quantile:
            for unit in self.forecast_time_units:
                result.append(pd.concat([forecast_quantile[target_name][unit.value],
                                         forecast[target_name][unit.value]]))
        result = pd.concat(result)
        result = result[(result['target_end_date'] >= self.forecast_date)
                     & (result['target_end_date'] <= self.forecast_time_range[-1])]

        result['location'] = str(self.fips)
        result['forecast_date'] = self.forecast_date
        result['forecast_date'] = result['forecast_date'].dt.strftime('%Y-%m-%d')
        result = result[~result['target'].apply(lambda s: 'wk ahead inc hosp' in s)]

        self.result = result[COLUMNS]

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
        print(om.forecast_date)
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


def generate_us_result(targets=[Target.CUM_DEATH],
                       units=[ForecastTimeUnit.DAY, ForecastTimeUnit.WK],
                       state_results=None,
                       state_results_path=None):
    """
    Aggregates State level forecast to US level.

    Parameters
    ----------
    targets: list
        List of Target objects as the forecast targets to include for US
        level.
    units: list
        List of forecast time unit as forecast time units to include for US
        level.
    state_results: pd.DataFrame
        Table that contains the forecast at states level (all US states).
        Contains columns:

    results: pd.DataFrame
        Results read from
    """
    # read US cumulative/incident death and
    dates, cumulative_death, incident_death = load_us_level_death_data()

    if state_results is None:
        if state_results_path is not None:
            state_results = pd.read_csv(state_results_path, dtype='str')
        else:
            raise ValueError('Neither State forecast table nor path of the '
                             'States forecast is provided.')

    state_results['value'] = state_results['value'].astype(float)
    target_key_words = [f'{unit.value} ahead {target.value}' for
                        unit in units for target in targets]

    # get States forecast quantiles for given targets
    with_right_target = state_results['target'].str.contains('|'.join(target_key_words))
    with_quantiles = state_results['type'] == 'quantile'
    results_with_target = state_results[with_right_target & with_quantiles]

    target_values = results_with_target.groupby(['target', 'location'])['value'].apply(
        list).reset_index().groupby('target')['value'].apply(
        lambda l: np.array(list(l)))


    us_result_quantiles = defaultdict(list)
    for target in target_values.index:
        # For each target, e.g. 1 day ahead cum death, collect each State's
        # forecast quantiles and draw random sample from the distribution.
        samples = list()
        for n in range(len(target_values.loc[target])):
            ppf = scipy.interpolate.interp1d(QUANTILES, target_values.loc[target][n])
            rvs = random_sample_from_ppf(np.vectorize(ppf),
                                         size=5000,
                                         bounds=(QUANTILES[0], QUANTILES[-1]))
            samples.append(rvs)
        samples = np.array(samples)
        # Get sample of US forecast as sum of random samples of States' forecast
        sum_of_samples = samples.sum(axis=0)
        us_result_quantiles['value'].append(ppf_from_data(sum_of_samples)(
            QUANTILES))
        us_result_quantiles['target'].append(target)

    us_result_quantiles = pd.DataFrame(us_result_quantiles).set_index('target')
    num_of_targets = us_result.shape[0]
    us_result_quantiles = us_result_quantiles.explode('value').reset_index()
    us_result_quantiles['quantile'] = np.tile(QUANTILES, num_of_targets)
    us_result_quantiles['type'] = 'quantile'
    us_result_quantiles['quantile'] = us_result['quantile'].apply(lambda v: '%.3f' % v)

    us_result_point = us_result_quantiles[us_result_quantiles['quantile'] == '0.500'][
        ['target', 'value']].copy()
    us_result_point['type'] = 'point'
    us_result = pd.concat([us_result_quantiles, us_result_point])
    us_result = pd.merge(us_result,
                         state_results[['target', 'target_end_date',
                                        'forecast_date']].drop_duplicates(),
                         on='target')
    us_result['value'] = us_result.apply(
        lambda r: np.maximum(cumulative_death[-1],
                             r.value).round() if 'cum death' in r.target
        else np.maximum(cumulative_death[-1], 0),
        axis=1)
    us_result['location'] = 'US'

    return us_result[COLUMNS]


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
    us_result = generate_us_result(state_results=results)
    results = pd.concat([results, us_result])
    results.to_csv(os.path.join(REPORT_FOLDER, f'{forecast_date}-{TEAM}-{MODEL}.csv'),
                   index=False)

    OutputMapper.generate_metadata()
