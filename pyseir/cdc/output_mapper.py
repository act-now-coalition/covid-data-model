import itertools
import os
import us
import scipy
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, date
from multiprocessing import Pool
from epiweeks import Week, Year
from string import Template
from pyseir import OUTPUT_DIR, load_data
from pyseir.utils import REF_DATE
from pyseir.cdc.utils import Target, ForecastTimeUnit, ForecastUncertainty, target_column_name
from pyseir.inference.fit_results import load_inference_result, load_mle_model
from pyseir.load_data import HospitalizationDataType
from statsmodels.nonparametric.kernel_regression import KernelReg


"""
This mapper maps current pyseir model output to match cdc format.

Output file should have columns:
- forecast_date: the date on which the submitted forecast data was made available in YYYY-MM-DD format
- target: Values in the target column must be a character (string) and have format "<day_num> day ahead <target_measure>"
          where day_num is number of days since forecast_date to each date in forecast time range. 
- target_end_date: end date of forecast in YYYY-MM-DD format.
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

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
REPORT_FOLDER = os.path.join(DIR_PATH, 'report')


TEAM = 'CovidActNow'
MODEL = 'SEIR_CAN'

# type of target measures
TARGETS = ['cum death', 'inc death', 'inc hosp']

# names of target measures that will be used to generate metadata
TARGETS_TO_NAMES = {'cum death': 'cumulative deaths',
                    'inc death': 'incident deaths',
                    'inc hosp': 'incident hospitalizations'}

DATE_FORMAT = '%Y-%m-%d'
# units of forecast target.
FORECAST_TIME_UNITS = ['day', 'wk']
# number of weeks ahead for forecast.
FORECAST_WEEKS_NUM = 4
# Default quantiles required by CDC.
QUANTILES = np.concatenate([[0.01, 0.025], np.arange(0.05, 0.95, 0.05), [0.975, 0.99]])
# Time of forecast, default date when this runs.
FORECAST_DATE = datetime.today() - timedelta(days=1)
# Next epi week. Epi weeks starts from Sunday and ends on Saturday.
#if forecast date is Sunday or Monday, next epi week is the week that starts
#with the latest Sunday.
if FORECAST_DATE.weekday() in (0, 6):
    NEXT_EPI_WEEK = Week(Year.thisyear().year, Week.thisweek().week)
else:
    NEXT_EPI_WEEK = Week(Year.thisyear().year, Week.thisweek().week + 1)
COLUMNS = ['forecast_date', 'location', 'location_name', 'target', 'type',
           'target_end_date', 'quantile', 'value']


class OutputMapper:
    """
    This mapper maps CAN SEIR model inference results to the format required
    for CDC model submission. For the given State FIPS code, it reads in the
    most up-to-date MLE inference (mle model + fit_results json file),
    and runs the model ensemble when fixing the parameters varied for model
    fitting at their MLE estimate. Quantile of forecast is then derived from
    the forecast ensemble weighted by the chi square obtained by fitting
    corresponding model to observed cases, deaths w/o hospitalizations,
    aiming to obtain the uncertainty associated with the prior distribution
    of the parameters not varied during model fitting and the (
    unknown) likelihood function (L(y_1:t|theta)). This for sure will
    underestimate the level of uncertainty of the forecast since it does not
    take into account the parameters varied for MLE inference.
    However, the error associated with the parameters for inference is
    generally small, so their variations are also relatively small.

    The output has the columns required for CDC model ensemble
    submission (check description of results). It currently supports daily
    and weekly forecast.

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
        daily forecasts.
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
        self.observations = self.load_observations()
        self.model = load_mle_model(self.fips)

        self.fit_results = load_inference_result(self.fips)

        self.errors = None
        self.result = None


    def load_observations(self):
        """
        Load observations based on type of target and unit.

        Returns
        -------

        """
        times, observed_new_cases, observed_new_deaths = \
            load_data.load_new_case_data_by_state(us.states.lookup(self.fips).name,
                                                  REF_DATE)
        dates = [timedelta(int(t)) + REF_DATE for t in times]

        hospital_times, hospitalizations, hospitalization_data_type = \
            load_data.load_hospitalization_data_by_state(us.states.lookup(self.fips).abbr,
                                                         REF_DATE)
        if hospital_times is not None:
            hospital_dates = [timedelta(int(t)) + REF_DATE for t in hospital_times]

        observations = defaultdict(dict)
        for target in self.targets:
            if target is Target.CUM_DEATH:
                observations[target.value] = pd.Series(observed_new_deaths.cumsum(),
                                                       index=pd.DatetimeIndex(dates).strftime(DATE_FORMAT))
            elif target is Target.INC_DEATH:
                smoothed_observed_new_deaths = self._smooth_observation(times, observed_new_deaths)
                observations[target.value] = pd.Series(smoothed_observed_new_deaths.clip(min=0),
                                                       index=pd.DatetimeIndex(dates).strftime(DATE_FORMAT))
            elif target is Target.INC_HOSP:
                if (hospital_times is not None) and \
                    (hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS):
                    observed_hospitalizations = np.append(hospitalizations[0],
                                                          np.diff(hospitalizations))
                    smoothed_observed_hospitalizations = self._smooth_observation(hospital_times,
                                                                                  observed_hospitalizations)
                    observations[target.value] = pd.Series(smoothed_observed_hospitalizations.clip(min=0),
                                                           index=pd.DatetimeIndex(hospital_dates).strftime(
                                                                 DATE_FORMAT))
                else:
                    observations[target.value] = None

        return observations


    def generate_forecast(self, start_date=None, align_with_observations=True):
        """
        Generates a forecast ensemble given the model ensemble.

        Parameters
        ----------
        model: list(SEIRModel) or NoneType
            List of SEIR models run under parameter sets randomly generated
            from the parameter prior distributions.

        Returns
        -------
        forecast_ensemble: dict(pd.DataFrame)
            Contains forecast of target within the forecast time window run by
            each model from the model ensemble. With "<num> <unit> ahead
            <target_measure>" as index, and corresponding value from each model
            as columns, where unit can be 'day' or 'wk' depending on the
            forecast_time_units.
        """
        start_date = start_date or datetime.strptime(self.observations['cum death'].index[-1], DATE_FORMAT)
        forecast_days_since_ref_date = list(range((start_date - REF_DATE).days,
                                                  (self.forecast_time_range[-1] - REF_DATE).days + 1))
        forecast_dates = [timedelta(t) + REF_DATE for t in forecast_days_since_ref_date]

        forecast_given_time_range = \
            lambda forecast, t_list: np.interp(forecast_days_since_ref_date,
                                               [self.fit_results['t0'] + t for t in t_list], forecast)

        forecast = {}
        for target in self.targets:
            if target is Target.INC_DEATH:
                forecast[target.value] = forecast_given_time_range(self.model.results['total_deaths_per_day'],
                                                                   self.model.t_list)

            elif target is Target.INC_HOSP:
                forecast[target.value] = forecast_given_time_range(np.append([0],
                                                                   np.diff(self.model.results['HGen_cumulative']
                                                                         + self.model.results['HICU_cumulative'])),
                                                                   self.model.t_list)

            elif target is Target.CUM_DEATH:
                forecast[target.value] = forecast_given_time_range(self.model.results['D'], self.model.t_list)

            else:
                raise ValueError(f'Target {target} is not implemented')


            forecast[target.value] = pd.DataFrame({'value': forecast[target.value].clip(min=0),
                                                   'target_end_date': forecast_dates,
                                                   'forecast_days': [(t - start_date).days for t in forecast_dates]},
                                                   index=pd.DatetimeIndex(forecast_dates).strftime(DATE_FORMAT))
            forecast[target.value]['target_end_date'] = forecast[target.value]['target_end_date'].astype('datetime64['
                                                                                                        'D]')
            forecast[target.value]['type'] = 'point'

        if align_with_observations:
            forecast = self.align_forecast_with_observations(forecast)

        return forecast

    def align_forecast_with_observations(self, forecast, ref_observation_date=None):
        """

        """

        ref_observation_date = ref_observation_date or self.observations['cum death'].index[-1]

        shifted_forecast = forecast.copy()
        # align with observations
        for target in self.targets:
            if self.observations[target.value] is not None:
                observation = self.observations[target.value]

                shifted_forecast[target.value]['value'] += \
                    (observation.loc[ref_observation_date]
                   - forecast[target.value]['value'].loc[ref_observation_date]).clip(min=0)

        return shifted_forecast


    def calculate_errors(self):
        """
        Collect distribution of absolute errors abs(y_t - y_t|y_t-1).

        Parameters
        ----------

        """
        self.errors = defaultdict(list)
        for date in self.observations['cum death'].index[:-1]:
            forecast = self.generate_forecast(start_date=datetime.strptime(date, DATE_FORMAT))
            for target in self.targets:
                if self.observations[target.value] is not None:
                    next_day = datetime.strftime(datetime.strptime(date, DATE_FORMAT) + timedelta(1), DATE_FORMAT)
                    if next_day in self.observations[target.value].index:
                        pred = forecast[target.value]['value'].loc[next_day]
                        observ = self.observations[target.value].loc[next_day]
                        self.errors[target.value].append(abs(pred - observ))

        return self.errors


    def _calculate_forecast_quantiles(self, forecast, error_std, h, quantiles):
        """
        Rescale forecast standard deviation by streching/shrinking the forecast
        distribution (gaussian) around the mean.
        Currently supports two approaches:
        - default: no adjustment
        - naive: rescale the standard deviation by factor (days_ahead  **
                 0.5)

        Parameters
        ----------
        data: np.array or list
            Data sample from the distribution.
        h: int or float
            Time step of forecast

        Returns
        -------
          :  np.array
            Data after the adjustment.
        """
        if self.forecast_uncertainty is ForecastUncertainty.DEFAULT:
            return np.maximum(forecast + scipy.stats.norm(loc=0, scale=error_std).ppf(quantiles), 0)
        elif self.forecast_uncertainty is ForecastUncertainty.NAIVE:
            return np.maximum(forecast + scipy.stats.norm(loc=0, scale=error_std*h**0.5).ppf(quantiles), 0)
        else:
            raise ValueError(f'forecast accuracy adjustment {self.forecast_uncertainty} is not implemented')

    def _smooth_observation(self, time, observations):
        """

        """
        kr = KernelReg(observations, time, 'c')
        smoothed, _ = kr.fit(time)
        return smoothed


    def generate_forecast_quantiles(self, forecast, quantiles=QUANTILES):
        """
        Runs forecast of a target with given model.

        Parameters
        ----------
        model: SEIRModel
            SEIR model to run the forecast.
        target: Target
            The target to forecast.
        unit: ForecastTimeUnit
            Time unit to aggregate the forecast.

        Returns
        -------
        target_forecast: np.array
            Forecast of target at given unit (daily or weekly), with shape (
            len(self.forecast_time_range),)
        """
        forecast_quantiles = dict()
        for target_name in forecast:
            if target_name in self.errors:
                error_std = np.std(self.errors[target_name])
            else:
                error_std = np.sqrt(forecast[target_name]['value'].iloc[0])

            forecast_quantiles[target_name] = forecast[target_name].apply(
                lambda r: self._calculate_forecast_quantiles(forecast=r.value,
                                                             error_std=error_std,
                                                             h=r.forecast_days,
                                                             quantiles=quantiles), axis=1).rename('value')

            forecast_quantiles[target_name] = pd.DataFrame(forecast_quantiles[target_name].explode())
            forecast_quantiles[target_name]['target_end_date'] = forecast[target_name]['target_end_date']
            forecast_quantiles[target_name]['quantile'] = np.tile(self.quantiles, forecast[target_name].shape[0])
            forecast_quantiles[target_name]['type'] = 'quantile'

        return forecast_quantiles


    def run(self):
        """
        Runs forecast ensemble. Results contain quantiles of
        the forecast targets and saves results to csv file.

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
                df = pd.concat([forecast_quantile[target_name], forecast[target_name]])
                if unit is ForecastTimeUnit.DAY:
                    num_of_units = [(datetime.date(datetime.strptime(str(t).partition('T')[0], DATE_FORMAT))
                                   - datetime.date(self.forecast_date)).days for t in df['target_end_date'].values]

                elif unit is ForecastTimeUnit.WK:
                    # Weekly report use forecast on Saturdays
                    df['is_saturday'] = df['target_end_date'].apply(lambda t: t.weekday() == 5)

                    if Target(target_name) is Target.CUM_DEATH:
                        df = df[df['is_saturday']]
                    else:
                        # if target is incident death or hospitalization, get the cumulative sum
                        df = df[df['type'] == 'point']
                        df['week'] = df['target_end_date'].apply(lambda t: Week.fromdate(t).week)
                        df['week'] -= df['week'].min() - 1 # shift first week to week 1
                        df = df.groupby('week').agg({'target_end_date': np.max,
                                                     'value': sum}).reset_index()
                        df_quantile = df.copy()

                        if target_name in self.errors:
                            error_std = np.std(self.errors[target_name])
                        else:
                            error_std = np.sqrt(df['value'].iloc[0])

                        # assume error has poisson distribution with mean as the forecast value adjusted by forecast
                        # forward days
                        df_quantile = df_quantile.set_index('target_end_date').apply(
                            lambda r: self._calculate_forecast_quantiles(forecast=r.value,
                                                                         error_std=error_std,
                                                                         h=r.week * 7,
                                                                         quantiles=self.quantiles),
                            axis=1).rename('value')
                        df_quantile = pd.DataFrame(df_quantile.explode()).reset_index()
                        df_quantile['quantile'] = np.tile(self.quantiles, df.shape[0])
                        df_quantile['type'] = 'quantile'

                        df['type'] = 'point'
                        df = pd.concat([df, df_quantile])

                    num_of_units = [(datetime.strptime(str(t).partition('T')[0], DATE_FORMAT) -
                                     self.forecast_date).days // 7 + 1 for t in df['target_end_date'].values]

                df['target'] = list(target_column_name(num_of_units, Target(target_name), unit))
                result.append(df)

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
    def run_for_fips(cls, fips):
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
        om = cls(fips)
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


def run_all(parallel=False):
    """
    Prepares inference results for all whitelist States for CDC model
    ensemble submission.

    Parameters
    ----------
    parallel: bool
        Whether to run multiprocessing.
    """
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
            result = OutputMapper.run_for_fips(fips)
            results.append(result)

    forecast_date = FORECAST_DATE.strftime(DATE_FORMAT)

    results = pd.concat(results)
    results = results[COLUMNS].sort_values(COLUMNS)
    results.to_csv(os.path.join(REPORT_FOLDER, f'{forecast_date}-{TEAM}-{MODEL}.csv'),
                   index=False)

    OutputMapper.generate_metadata()
