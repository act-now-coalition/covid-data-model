import pandas as pd
import numpy as np
from enum import Enum

class ErrorType(Enum):
    RELATIVE_ERROR = 'relative_error'
    PERCENTAGE_ABS_ERROR = 'percentage_abs_error'
    SYMMETRIC_ABS_ERROR = 'symmetric_abs_error'
    BOUNDED_RELATIVE_ERROR = 'bounded_relative_error'
    RMSE = 'rmse'
    NRMSE = 'nrmse'
    ERROR = 'error'
    ABS_ERROR = 'abs_error'

class Measure(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    CI_95 = 'ci_95'
    CI_68 = 'ci_68'
    VAR = 'var'
    STD = 'std'

def error_type_to_meaning(error_type):
    """
    Translate error type string to corresponding meaning.

    Parameters
    ----------
    error_type: str
        Type of prediction error.
    """
    error_type = ErrorType(error_type)
    if error_type is ErrorType.RMSE:
        return 'root of mean squared error'
    if error_type is ErrorType.NRMSE:
        return 'normalized root of mean squared error'
    if error_type is ErrorType.ERROR:
        return 'error'
    if error_type is ErrorType.RELATIVE_ERROR:
        return 'percentage error'
    if error_type is ErrorType.PERCENTAGE_ABS_ERROR:
        return 'Absolute percentage error'
    if error_type is ErrorType.SYMMETRIC_ABS_ERROR:
        return 'Absolute error normalized by sum of prediction and observation'
    if error_type is ErrorType.ABS_ERROR:
        return 'Absolute error'


class TimeSeriesMetrics:
    """
    Calculates metrics of timeseries. Currently supporting calculation of
    error of prediction of following types:
    - rmse: root of mean squared error
    - nrmse: normalized root of mean squared error
    - error: residuals
    - relative_error: percentage error
    - percentage_abs_error: absolute percentage error
    - symmetric_abs_error: absolute error normalized by sum of prediction
                           and observation
    - abs_error: absolute error
    """

    def __init__(self):
        return

    def flatten_values(self, series):
        """
        Flatten timeseries into array.
        """
        if isinstance(series, pd.Series) or isinstance(series, pd.DataFrame):
            return series.values.ravel()
        return series


    def calculate_error(self, y_true, y_pred, error_type, missing_observation=np.nan):
        """
        Calculate error of predictions given type of error.

        Parameters
        ----------
        y_true: pd.Series or np.array
            Observations
        y_pred: pd.Series or np.array
            Predictions
        error_type: str
            Type of prediction error to calculate, should be interpretable by ErrorType.
        missing_observation: float
            Value that indicates the observation is missing.

        Returns
        -------
        error: np.array or float
            Prediction error.
        """
        y_true = self.flatten_values(y_true)
        y_pred = self.flatten_values(y_pred)

        valid_observations = np.where(y_true != missing_observation)
        invalid_observations = np.where(y_true == missing_observation)
        error_type = ErrorType(error_type)

        if error_type is ErrorType.RELATIVE_ERROR:
            error = (y_pred - y_true) / y_true
        elif error_type is ErrorType.ERROR:
            error = y_pred - y_true
        elif error_type is ErrorType.ABS_ERROR:
            error = np.abs(y_pred - y_true)
        elif error_type is ErrorType.PERCENTAGE_ABS_ERROR:
            error = np.abs(y_pred - y_true)/y_true
        elif error_type is ErrorType.SYMMETRIC_ABS_ERROR:
            error = np.abs(y_pred - y_true)/np.abs(y_true + y_pred)
        elif error_type is ErrorType.RMSE:
            error = np.sqrt(np.square(np.subtract(y_true[valid_observations],
                                                  y_pred[valid_observations])).mean())
        elif error_type is ErrorType.NRMSE:
            error = np.sqrt(np.square(np.subtract(y_true[valid_observations],
                                                  y_pred[valid_observations])).mean()) / y_true[valid_observations].mean()

        if error_type not in [ErrorType.RMSE, ErrorType.NRMSE]:
            error[invalid_observations] = np.nan

        return error


    def calculate_error_measure(self, error, measure):
        """
        Calculate measure of prediction error, including: mean, median,
        variance, standard deviation, 95% confidence interval or 68%
        confidence interval.

        Parameters
        ----------
        error: np.array
            Prediction errors
        measure: str
            Name of the measure, should be interpretable by Measure class.

        Returns
        -------
          :  float or np.array
            Measure of the prediction errors.
        """
        measure = Measure(measure)

        if measure is Measure.MEAN:
            return np.mean(error)
        elif measure is Measure.MEDIAN:
            return np.median(error)
        elif measure is Measure.VAR:
            return np.var(error)
        elif measure is Measure.STD:
            return np.std(error)
        elif measure is Measure.CI_95:
            return (np.quantile(error, (0.025, 0.975)))
        elif measure is Measure.CI_68:
            return (np.quantile(error, (0.16, 0.84)))
