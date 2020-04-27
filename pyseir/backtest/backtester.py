import logging
import matplotlib.backends.backend_pdf
import us
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from multiprocessing import Pool
from functools import partial
from pyseir.utils import REF_DATE
from pyseir import load_data
from pyseir.load_data import HospitalizationDataType
from pyseir.inference.model_fitter import ModelFitter
from pyseir.utils import get_run_artifact_path, RunArtifact
from pyseir.backtest.timeseries_metrics import TimeSeriesMetrics, error_type_to_meaning


class Backtester:
    """
    Run backtest for a given fips. Backtest blinds last n days of
    observations (new cases, new deaths or hospitalizations) and calculates
    error of prediction in prediction_window_size days since last un-blinded
    observation.
    Errors can be calculated for each ith day of prediction where 1 <= i <=
    prediction_window_size or for the entire prediction time window,
    depending on type of errors.
    Prediction errors are calculated as number of days to blind observations
    increases until it reaches max_observation_days_blinded.
    To remove unwanted feature (smooth) of the time series of the
    observation data, one can set window of ts_rolling_args greater than one.
    To accelorate backtesting, one can set num_days_per_step greater than 1.
    However it is recommended that num_days_per_step <= ts_rolling_args[
    'window'] to ensure a good coverage of the time series.

    Attributes
    ----------
    observations: pd.DataFrame
        Contains observations for given fips codes, with columns:
            - new_cases: float, observed new cases
            - new_deaths: float, observed new deaths
            - hospitalizations: float, observed hospitalizations
                                and dates of observation as index.
        and dates of observations as index.
    backtest_results: pd.DataFrame
        Contains errors of predictions by days of prediction and type
        of error. With columns:
            - observation_type: str, type of observations, should be among:
                                'new_cases', 'new_deaths', or 'hospitalizations'
            - error_type: str, type of forecasting errors.
            - error: float, value of corresponding type of error
            - days_of_forecast: int, number of days of forecast since latest
                                available observation.
            - observation_end_date: Datetime, last day before blinded
                                    observations.
    historical_predictions: pd.DataFrame
        Contains predictions given historial observation timeseries.
        With columns:
            - dates: Datatime, dates of prediction
            - new_cases: float, predicted new cases
            - new_deaths: float, predicted new deaths
            - hospitalizations: float, predicted hospitalizations
            - observation_end_date: Datetime, last day before blinded
                                    observations.
            - observation_days_blinded: int, number of days that observations
                                        are blinded.

    Parameters
    ----------
    fips: str
        FIPS code of the state or county to run backtest.
    ts_rolling_args: dict
        Parameters that determine the moving average of timeseries of
        observation and prediction. Should follow the naming convention of
        pandas.Series.rolling (
        https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas
        .Series.rolling.html).
    ts_rolling_kernel_args: dict
        Parameters to pass to the rolling kernel function, should match
        ts_rolling_args['win_type']. For details, check description of
        win_type in: https://pandas.pydata.org/pandas-docs/version/0.23.4
        /generated/pandas.Series.rolling.html .
    prediction_window_size: int
        Size of time window of prediction to calculate prediction error.
    num_days_per_step: int
        Number of days of moving backward to blind the observations per step.
        For example, observations are blinded for 1, 3, 5 ... days if
        num_days_per_step is 2.
    max_observation_days_blinded: int
        Maximum number of days to blind the observations.
    error_types: str
        Type of prediction error to calculate, should be included in
        timeseries_metrics.ErrorType.
    ref_date: Datetime
        Reference start date.
    n_retries: int
        Number of times to retry model fitting.
    """
    def __init__(self,
                 fips,
                 ts_rolling_args={'window': 3,
                                  'min_periods': 1,
                                  'win_type': 'gaussian',
                                  'center': True},
                 ts_rolling_kernel_args={'std': 1},
                 num_days_per_step=3,
                 prediction_window_size=14,
                 max_observation_days_blinded=40,
                 error_types=['nrmse', 'rmse', 'relative_error',
                              'percentage_abs_error', 'symmetric_abs_error'],
                 ref_date=REF_DATE,
                 n_retries=3):

        self.fips = fips
        self.ts_rolling_args = ts_rolling_args
        self.ts_rolling_kernel_args = ts_rolling_kernel_args
        self.prediction_window_size = prediction_window_size
        self.max_observation_days_blinded = max_observation_days_blinded
        self.num_days_per_step = num_days_per_step
        self.error_types = error_types
        self.ref_date = ref_date
        self.n_retries = n_retries

        if self.num_days_per_step > self.ts_rolling_args['window']:
            logging.warning("num_days_per_step is greater than "
                            "ts_rolling_args['window'] which will make the "
                            "backtest skip part of the time series.")

        self.observations = None
        self.backtest_results = None
        self.historical_predictions = None


    @staticmethod
    def load_observations(fips=None, ref_date=REF_DATE):
        """
        Load observations (new cases, new deaths and hospitalizations) for
        given fips code.

        Parameters
        ----------
        fips: str
            FIPS code.
        ref_date: Datetime
            Reference start date.

        Returns
        -------
        observations: pd.DataFrame
            Contains observations for given fips codes, with columns:
            - new_cases: float, observed new cases
            - new_deaths: float, observed new deaths
            - hospitalizations: float, observed hospitalizatons
            and dates of observation as index.
        """

        observations = {}
        if len(fips) == 5:
            times, observations['new_cases'], observations['new_deaths'] = \
                load_data.load_new_case_data_by_fips(fips, ref_date)
            hospital_times, hospitalizations, hospitalization_data_type = \
                load_data.load_hospitalization_data(fips, t0=ref_date)
            observations['times'] = times.values
        elif len(fips) == 2:
            state_obj = us.states.lookup(fips)
            observations['times'], observations['new_cases'], observations['new_deaths'] = \
                load_data.load_new_case_data_by_state(state_obj.name, ref_date)
            hospital_times, hospitalizations, hospitalization_data_type = \
                load_data.load_hospitalization_data_by_state(state_obj.abbr, t0=ref_date)
            observations['times'] = np.array(observations['times'])

        observations['hospitalizations'] = np.full(observations['times'].shape[0], np.nan)
        if hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
            observations['hospitalizations'][hospital_times - observations['times'].min()] = np.diff(hospitalizations)
        elif hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
            observations['hospitalizations'][hospital_times - observations['times'].min()] = hospitalizations

        observation_dates = [ref_date + timedelta(int(t)) for t in observations['times']]
        observations = pd.DataFrame(observations, index=pd.DatetimeIndex(
            observation_dates)).dropna(axis=1, how='all')

        return observations

    def run_model_fitter_for_backtest(self, observation_days_blinded):
        """
        Run model fitter with observations blinded.

        Parameters
        ----------
        observation_days_blinded: int
            Last number of days to blind observations.

        Returns
        -------
        prediction: pd.DataFrame
            Predicted new cases, new deaths or hospitalizations given partially
            blinded observations, with dates of prediction as index.
            Contains:
            - new_cases: float, predicted new cases
            - new_deaths: float, predicted new deaths
            - hospitalizations: float, predicted hospitalizations (if
                                hospitalization data is available for given
                                fips).
        """

        mf = ModelFitter(self.fips)
        mf.times = self.observations['times'][:-observation_days_blinded]
        mf.observed_new_cases = self.observations['new_cases'].values[:-observation_days_blinded]
        mf.observed_new_deaths = self.observations['new_deaths'].values[:-observation_days_blinded]
        if mf.hospital_times is not None:
            mf.hospital_times = mf.hospital_times[mf.hospital_times <= mf.times.max()]
            mf.hospitalizations = mf.hospitalizations[:len(mf.hospital_times)]
            # when all hospitalization data has been blinded
            if mf.hospital_times.size == 0:
                mf.hospitalization_data_type = None

        mf.cases_stdev, mf.hosp_stdev, mf.deaths_stdev = mf.calculate_observation_errors()

        for n in range(self.n_retries):
            try:
                mf.fit()
                if mf.mle_model:
                    break
            except Exception as e:
                print(e)

        prediction = {}
        prediction['new_cases'] = mf.fit_results['test_fraction'] * \
                                  np.interp(self.observations['times'],
                                            mf.t_list + mf.fit_results['t0'],
                                            mf.mle_model.results['total_new_infections'])
        prediction['new_deaths'] = np.interp(self.observations['times'],
                                             mf.t_list + mf.fit_results['t0'],
                                             mf.mle_model.results['total_deaths_per_day'])

        if mf.hospitalization_data_type is not None:
            if mf.hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
                predicted_hosp = (mf.mle_model.results['HGen_cumulative'] + mf.mle_model.results['HICU_cumulative'])
                predicted_hosp = np.diff(predicted_hosp)
            elif mf.hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
                predicted_hosp = mf.mle_model.results['HGen'] + mf.mle_model.results['HICU']
            prediction['hospitalizations'] = np.interp(self.observations['times'],
                                                       mf.t_list + mf.fit_results['t0'],
                                                       predicted_hosp * mf.fit_results['hosp_fraction'])
        else:
            prediction['hospitalizations'] = None

        prediction = pd.DataFrame(prediction, index=self.observations.index).dropna(axis=1, how='all')

        return prediction


    def run_backtest(self):
        """
        Run backtest for a given fips. Backtest blinds last n days of observations (
        new cases, new deaths or hospitalizations) and calculates error of
        prediction in prediction_window_size days since last un-blinded
        observation. Errors can be calculated for ith day of prediction where 1
        <= i <= prediction_window_size or for the entire prediction time window,
        depending on type of errors.

        Returns
        -------
        backtest_results: pd.DataFrame
            Contains errors of predictions by days of prediction and type of
            error. With columns:
            - observation_type: str, type of observations, should be among:
                                'new_cases', 'new_deaths', or 'hospitalizations'
            - error_type: str, type of forecasting errors.
            - error: float, value of corresponding type of error
            - days_of_forecast: int, number of days of forecast since latest
                                available observation.
            - observation_end_date: Datetime, last day before blinded
                                    observations.
        historical_predictions: pd.DataFrame
            Contains predictions given historial observation timeseries.
            With columns:
            - dates: Datatime, dates of prediction
            - new_cases: float, predicted new cases
            - new_deaths: float, predicted new deaths
            - hospitalizations: float, predicted hospitalizations
            - observation_end_date: Datetime, last day before blinded
                                    observations.
            - observation_days_blinded: int, number of days that observations
                                        are blinded.
        """
        tsm = TimeSeriesMetrics()
        backtest_results = list()
        historical_predictions = list()

        for d in np.arange(1, self.max_observation_days_blinded + 1, self.num_days_per_step):
            observation_end_date = self.ref_date + timedelta(int(self.observations['times'].values[-d]))
            # record predictions
            prediction = self.run_model_fitter_for_backtest(observation_days_blinded=d)
            prediction['observation_end_date'] = observation_end_date
            prediction['observation_days_blinded'] = d

            # record back test errors
            backtest_record = defaultdict(list)

            moving_average = lambda s: s.rolling(**self.ts_rolling_args).mean(
                **self.ts_rolling_kernel_args)[-d:][:self.prediction_window_size]

            for observation_type in ['new_cases', 'new_deaths', 'hospitalizations']:
                if observation_type in self.observations:
                    for error_type in self.error_types:
                        error = tsm.calculate_error(
                            moving_average(self.observations[observation_type]),
                            moving_average(prediction[observation_type]),
                            error_type=error_type)

                        if error_type in ['rmse', 'nrmse']:
                            error = np.array([error])

                        backtest_record['observation_type'].extend([observation_type] * error.shape[0])
                        backtest_record['error_type'].extend([error_type] * error.shape[0])
                        backtest_record['observation_end_date'].extend([observation_end_date] * error.shape[0])
                        backtest_record['error'].extend(list(error))
                        if error_type in ['rmse', 'nrmse']:
                            backtest_record['days_of_forecast'].append(min(self.prediction_window_size, d))
                        else:
                            backtest_record['days_of_forecast'].extend(list(range(1, error.shape[0] + 1)))

            backtest_results.append(pd.DataFrame(backtest_record))
            historical_predictions.append(prediction.reset_index().rename(columns={'index': 'dates'}))

        self.backtest_results = pd.concat(backtest_results)
        self.historical_predictions = pd.concat(historical_predictions)

        return self.backtest_results, self.historical_predictions


    def plot_backtest_results(self, backtest_results=None, pdf=None):
        """
        Plot backtest results.

        Parameters
        ----------
        backtest_results: pd.DataFrame
            Contains errors of predictions by days of prediction and type of
            error. With columns:
            - observation_type: str, type of observations, should be among:
                                'new_cases', 'new_deaths', or 'hospitalizations'
            - error_type: str, type of forecasting errors.
            - error: float, value of corresponding type of error
            - days_of_forecast: int, number of days of forecast since latest
                                available observation.
            - observation_end_date: Datetime, last day before blinded observations.
        pdf: matplotlib.backends.backend_pdf
            Pdf object to save the plot.
        """

        if  backtest_results is None:
            backtest_results =self.backtest_results

        for observation_type in backtest_results.observation_type.unique():
            for error_type in backtest_results.error_type.unique():
                df = backtest_results[(backtest_results.error_type == error_type)
                                    & (backtest_results.observation_type == observation_type)]
                if error_type not in ['rmse', 'nrmse']:
                    fig, axes = plt.subplots(nrows=int(np.ceil(df.days_of_forecast.max() / 3)),
                                             ncols=3,
                                             figsize=(18, 10))
                    for d, ax in list(zip(df.days_of_forecast.unique(), np.ravel(axes))):
                        df[df.days_of_forecast == d].plot('observation_end_date', 'error', ax=ax)
                        ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
                        ax.set_title('%d days prediction' % d)
                        ax.set_xlabel('date of last observation')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.subplots_adjust(wspace=0.2, hspace=2)
                    plt.suptitle(f'{observation_type}\n{error_type_to_meaning(error_type)}', fontsize=15)
                else:
                    plt.figure()
                    df.drop_duplicates().plot(x='observation_end_date',
                                              y='error', kind="line",
                                              label=error_type)
                    plt.title(f'{observation_type}\n'
                              f'{error_type_to_meaning(error_type)}\n'
                              f'{backtest_results.days_of_forecast.max()} days prediction')
                    plt.legend()
                    plt.xlabel('date of last observation')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                if pdf is not None:
                    pdf.savefig()

    def plot_historical_predictions(self, historical_predictions=None, observations=None, pdf=None):
        """
        Plot predictions made given historial timeseries of observations.

        Parameters
        ----------
        historical_predictions: pd.DataFrame
            Contains predictions given historial observation timeseries.
            With columns:
            - dates: Datatime, dates of prediction
            - new_cases: float, predicted new cases
            - new_deaths: float, predicted new deaths
            - curret_hosp: float, predicted hospitalizations
            - observation_end_date: Datetime, last day before blinded observations.
            - observation_days_blinded: int, number of days that observations
                                        are blinded.
        observations: pd.DataFrame
            Contains most uptodate observations (new cases, new deaths and
            hospitalizations).
        pdf: matplotlib.backends.backend_pdf
            Pdf object to save the plot.
        """
        if historical_predictions is None:
            historical_predictions = self.historical_predictions
        if observations is None:
            observations = self.observations

        for observation_type in ['new_cases', 'new_deaths', 'hospitalizations']:
            if observation_type in historical_predictions.columns:
                fig, ax = plt.subplots()
                sns.lineplot(x='dates', y=observation_type, hue='observation_days_blinded',
                             data=historical_predictions,
                             palette='cool', **{'alpha': 0.5})
                sns.lineplot(observations.index,
                             observations[observation_type].values,
                             color='k', label='observed ' + observation_type)

                plt.yscale('log')
                plt.ylim(bottom=1)
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                if pdf is not None:
                    pdf.savefig()

    def generate_report(self):
        """
        Generate pdf report of backtesting results.
        """
        output_path = get_run_artifact_path(self.fips, 'backtest_result')
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
        self.plot_backtest_results(self.backtest_results, pdf)
        self.plot_historical_predictions(self.historical_predictions, self.observations, pdf)
        pdf.close()

    def run(self):
        """
        Run backtest for given fips and generates pdf report.
        """
        self.observations = self.load_observations(self.fips, self.ref_date)
        self.backtest_results, self.historical_predictions = self.run_backtest()
        self.generate_report()

    @classmethod
    def run_for_fips(cls, fips, kwargs=None):
        """
        Run backtest for given fips and output reports.
        Backtest blinds n days of observations (cases, deaths or
        hospitalizations) and calculates error of prediction in n +
        prediction_window_size days. Errors can be calculated
        for ith day of prediction where 1 <= i <= prediction_window_size or for
        the entire prediction time window, depending on type of errors.

        Parameters
        ----------
        fips: str
            State's or county's FIPS code.
        kwargs: dict
            Parameters passed to Backtester other than 'fips', should be within:
            - ts_rolling_args: dict, parameters for pandas.Series.rolling,
              for defails, check
              https://pandas.pydata.org/pandas-docs/version/0.23.4
              /generated/pandas.Series.rolling.html
            - ts_rolling_kernel_args:  dict, parameters for
              pandas.Series.rolling window kernel, for defails,
              check description of win_type in:
              https://pandas.pydata.org/pandas-docs/version/0
              .23.4/generated/pandas.Series.rolling.html
            - prediction_window_size
            - max_observation_days_blinded
            - error_types
            - ref_date
            - n_retries
        """
        kwargs = kwargs or {}
        backtester = cls(fips=fips, **kwargs)
        backtester.run()


def run_for_fips_list(fips=['06', '36',
                           '06075', '06073', '36047',
                           '36081', '36005', '13121'],
                     kwargs=None):
    """
    Run backtest for given list of fips.
    Default fips list contains codes of states or counties with high covid
    burden:
    06 - CA, 36 - NY, 06075 - San Francisco County, 06073 - Log Angeles County,
    36047 - Kings County, 36081 - Bronx County, 13121 - Fulton County.

    Parameters
    ----------
    fips: str or list(str)
        FIPS codes to run backtest for.
    kwargs: dict
        kwargs passed for backtet, should be within:
        - ts_rolling_args: dict, parameters for pandas.Series.rolling,
          for defails, check: https://pandas.pydata.org/pandas-docs/version/0
          .23.4/generated/pandas.Series.rolling.html
        - ts_rolling_kernel_args:  dict, parameters for
          pandas.Series.rolling window kernel, for defails, check description
          of win_type in: https://pandas.pydata.org/pandas-docs/version/0.23
          .4/generated/pandas.Series.rolling.html
        - prediction_window_size
        - max_observation_days_blinded
        - num_days_per_step
        - error_types
        - ref_date
        - n_retries
    """

    kwargs = kwargs or {}
    p = Pool()
    p.map(partial(Backtester.run_for_fips, kwargs), fips)
    p.close()
