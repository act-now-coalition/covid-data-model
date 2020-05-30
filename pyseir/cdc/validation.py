from datetime import timedelta
from pyseir.cdc.utils import target_column_name
from pyseir.cdc.parameters import (FORECAST_DATE, Target, ForecastTimeUnit,
                                   TEAM, MODEL, DATE_FORMAT,
                                   TARGETS, FORECAST_TIME_UNITS,
                                   TARGETS_TO_NAMES)
from pyseir.cdc.output_mapper import REPORT_FOLDER
import scipy
from pyseir.cdc.output_mapper import OutputMapper, run_all as run_output_mapper
import us
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
from matplotlib.backends import backend_pdf
from pyseir import load_data
from pyseir.cdc.parameters import Target, ForecastTimeUnit
from pyseir.cdc.utils import load_and_aggregate_observations
from pyseir.cdc.output_mapper import (OutputMapper, REPORT_FOLDER, TEAM,
                                      MODEL, FORECAST_DATE, DATE_FORMAT)


class Validation:
    """
    Currently supports validation of daily forecast.
    Mainly calculates the interval scores of a state forecast by comparing the backtesting forecast
    with the observations.
    Interval scores are calculated with the interval scoring approach from Reichlab:
    reference: https://arxiv.org/pdf/2005.12881.pdf
    """

    def __init__(self, fips, forecast_date, pdf=None, forecast=None):
        self.fips = fips
        self.forecast_date = forecast_date
        self.targets = [Target(t) for t in TARGETS]
        self.forecast_time_units = [ForecastTimeUnit(u) for u in FORECAST_TIME_UNITS]
        self.forecast = forecast or pd.read_csv(
            f'{REPORT_FOLDER}/{datetime.strftime(forecast_date, DATE_FORMAT)}_{TEAM}_{MODEL}_{fips}.csv')
        self.forecast['quantile'] = self.forecast['quantile'].astype(float)
        self.observations = load_and_aggregate_observations(fips, units=self.forecast_time_units, targets=self.targets)
        self.alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.pdf = pdf

        self.quantile_func = None
        self.interval_scores = None

    def extract_quantiles(self):
        """

        """
        quantile = defaultdict(dict)
        for target in self.targets:
            if self.observations[target.value][ForecastTimeUnit.DAY.value] is not None:
                last_observation_date = \
                    datetime.strptime(self.observations[target.value][ForecastTimeUnit.DAY.value].index[-1],
                                      DATE_FORMAT)
                for target_end_date in self.forecast['target_end_date']:
                    if datetime.strptime(target_end_date, DATE_FORMAT) <= last_observation_date:
                        target_name = f'day ahead {target.value}'
                        df = self.forecast[(self.forecast['target_end_date'] == target_end_date)
                                         & (self.forecast['target'].str.contains(target_name))][['quantile', 'value']]

                        quantile[target.value][target_end_date] = \
                                scipy.interpolate.interp1d(df['quantile'], df['value'])


        self.quantile_func = dict(quantile)
        return self.quantile_func


    def interval_score(self, alpha, quantile_func, y):
        """
        This score can be interpreted heuristically as a measure of distance
        between the predictive
        distribution and the true observation, where the units are similar to
        those of absolute error,
        on the natural scale of the data.

        It is negatively oriented, meaning a lower score is better.

        See Reich Lab Forecasting Hub preprint "Evaluating epidemic forecasts in
        an interval format".

        Parameters
        ----------
        alpha: float
          Prediction int used to calculate a central (1-alpha)x100% prediction
          interval. So an alpha of .1
          would be the 90% prediction interval and would use the .05 and .95
          quantiles are predictions.
        quantile_func: callable
          The function to draw quantiles. Can be anything that has a
          self.quantile method that returns
          a quantile when passed a value in [0,1]
        y: float
          The observed value

        Returns
        -------
        score: float
          The interval score for a given alpha, predictive distribution F,
          and observed Y.
        """
        # The score has two main parts. A score for how sharp your estimate is (
        # tighter is better) and a
        # penalty if the observed is outside the estimate. The penalty is
        # proportional to the distance
        # between the estimate and your estimate.

        # First Part: Sharpness Score
        lower_quantile = alpha / 2
        upper_quantile = 1 - (alpha / 2)
        upper_estimate = quantile_func(upper_quantile)
        lower_estimate = quantile_func(lower_quantile)

        range_score = upper_estimate - lower_estimate

        # Second Part: Penality for Observed Outside Predicted Range
        if y >= lower_estimate and y <= upper_estimate:  # Within Range
            penalty = 0
        elif y < lower_estimate:  # Below Range
            penalty = (2 / alpha) * (lower_estimate - y)
        else:  # Above Range
            penalty = (2 / alpha) * (y - upper_estimate)

        return range_score + penalty


    def weighted_interval_score(self, alphas, quantile_func, y):
        """
        The weighted interval score as described in Reich COVID-19 Forecast Hub.
        We are following the nomenclature in the paper for ease of comparison

        Parameters
        ----------
        alphas: list[float]
          A list of alphas to include in central predictive interval tests. All
          alphas must be in (0,1).
          Do not include the median (alpha=1). If you pass in an empty list we
          return the point forecast
          interval score, which is just the absolute error.
        F: pandas.Series
          The function to draw quantiles. Can be anything that has a
          self.quantile method that returns
          a quantile when passed a value in [0,1]
        y: float
          The observed value

        Returns
        -------
        score: float
          The weighted interval score for a given alpha, predictive distribution
          F, and observed Y.
        """

        # Calculate the absolute error of the median
        w_0 = 1 / 2
        score_median = w_0 * 2 * abs(y - quantile_func(.5))

        # Calculate the interval scores for each provided alpha
        # Note that the weight is proportial to the alpha/2
        score_individual_intervals = [(a / 2) * self.interval_score(a, quantile_func, y) for a in
                                      alphas]

        # Calculate the average of all of them
        summed_scores = score_median + sum(score_individual_intervals)
        num_of_intervals = 1 + len(alphas)
        return summed_scores / num_of_intervals


    def calculate_interval_scores(self):
        """

        """
        interval_scores = defaultdict(dict)
        for target in self.targets:
            observed = self.observations[target.value][ForecastTimeUnit.DAY.value]
            if observed is not None:
                interval_scores[target.value]['date'] = list()
                interval_scores[target.value]['interval score'] = list()
                for date in observed.index:
                    if datetime.strptime(date, DATE_FORMAT) >= self.forecast_date:
                        score = self.weighted_interval_score(
                            self.alphas, self.quantile_func[target.value][date], observed.loc[date])
                        interval_scores[target.value]['date'].append(datetime.strptime(date, DATE_FORMAT))
                        interval_scores[target.value]['interval score'].append(score)

                interval_scores[target.value] = pd.DataFrame(interval_scores[target.value]).set_index('date')

        self.interval_scores = dict(interval_scores)
        return self.interval_scores


    def plot(self, interval_scores=None):
        """

        """
        interval_scores = interval_scores or self.interval_scores
        plt.figure(figsize=(10, 10))
        for n, target_name in enumerate(interval_scores.keys()):
            plt.subplot(len(interval_scores.keys()), 2, n*2 + 1)
            observed = self.observations[target_name][ForecastTimeUnit.DAY.value]
            observed.index = pd.to_datetime(observed.index)
            plt.plot(observed.index, observed.values, label='observed')

            df = self.forecast[self.forecast['target'].str.contains(f'day ahead {target_name}')]
            df['target_end_date'] = pd.to_datetime(df['target_end_date'])
            plt.plot(df[df['quantile'] == 0.5]['target_end_date'],
                     df[df['quantile'] == 0.5]['value'],
                     label=f'{TARGETS_TO_NAMES[target_name]} forecast median',
                     color='orange')


            plt.axvline(x=self.forecast_date, label='date of backtesting forecast', color='0.5', linestyle=':')
            plt.xlim(xmin=self.forecast_date - timedelta(days=7))
            plt.ylabel(TARGETS_TO_NAMES[target_name])

            plt.fill_between(x=df[df['quantile'] == 0.5].sort_values('target_end_date')['target_end_date'],
                                    y1=df[df['quantile'] == 0.025].sort_values('target_end_date')['value'],
                                    y2=df[df['quantile'] == 0.975].sort_values('target_end_date')['value'],
                                    label=f'CI_95',
                                    facecolor='b',
                                     alpha=0.2)
            plt.xticks(rotation=45)

            plt.legend()

            plt.subplot(len(interval_scores.keys()), 2, n * 2 + 2)
            plt.plot(interval_scores[target_name].index,
                     interval_scores[target_name]['interval score'],
                     label=f'{TARGETS_TO_NAMES[target_name]} interval score')
            plt.xlabel('forecast date')
            plt.ylabel('\n'.join(['interval score of', TARGETS_TO_NAMES[target_name] + ' forecast']))
            plt.legend()
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.suptitle('/'.join([us.states.lookup(self.fips).name, self.fips]))
        plt.subplots_adjust(top=0.88)

        if self.pdf:
            self.pdf.savefig()


    def run(self):
        self.quantile_func = self.extract_quantiles()
        self.interval_scores = self.calculate_interval_scores()
        self.plot()


def run_validation(window=7, forecast_date=FORECAST_DATE, run_om=True):
    """

    """
    backtesting_forecast_date = forecast_date - timedelta(days=window)
    if run_om:
        run_output_mapper(mapper_kwargs = {'forecast_date': backtesting_forecast_date})

    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist['inference_ok'] == True]
    fips_list = list(df_whitelist['fips'].str[:2].unique())

    output_path = f'{REPORT_FOLDER}/validation_report_{datetime.strftime(forecast_date, DATE_FORMAT)}.pdf'
    pdf = backend_pdf.PdfPages(output_path)
    for fips in fips_list:
        v = Validation(fips=fips, forecast_date=backtesting_forecast_date, pdf=pdf)
        v.run()
    pdf.close()
