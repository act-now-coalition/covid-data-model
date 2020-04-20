import math
import numpy as np
import pandas as pd
from statsmodels.tsa.api import (
    adfuller, bds, coint, kpss, acf, q_stat
)
from statsmodels.stats import diagnostic
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import namedtuple
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import metrics
from scipy.stats import mstats
from scipy.stats import iqr
from empiricaldist import Cdf as CDF
from scipy import linalg

def flatten_values(series: pd.Series):
    if isinstance(series, pd.Series) or isinstance(series, pd.DataFrame):
        return series.values.ravel()
    return series

def _relative_error(y_true: pd.Series, y_pred: pd.Series):
    """ Relative Error """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    if benchmark is None or benchmark.is_integer():
        if benchmark == 0:
            seasonality = 1
        else:
            seasonality = benchmark
        error = y_true[seasonality:] - y_pred[seasonality:]
        naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]  
        return error / naive_forecast_error
    return (y_true - y_pred) / y_true

def _bounded_relative_error(y_true: pd.Series, y_pred: pd.Series):
    """ Bounded Relative Error """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    if benchmark is None or benchmark.is_integer():
        # If no benchmark prediction provided - use naive forecasting
        if benchmark == 0:
            seasonality = 1
        else:
            seasonality = benchmark
        error = y_true[seasonality:] - y_pred[seasonality:]
        abs_error = np.abs(error)
        naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]
        abs_error_bench = np.abs(naive_forecast_error)
    else:
        error = y_true[seasonality:] - y_pred[seasonality:]
        abs_error = np.abs(error)
        abs_error_bench = np.abs(y_true - benchmark)
    return abs_error / (abs_error + abs_error_bench)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Root Mean Squared Error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Root Mean Squared Error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.sqrt(
        metrics.mean_squared_error(y_true, y_pred)
    )

def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Normalized Root Mean Squared Error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Normalized Root Mean Squared Error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    _rmse = root_mean_squared_error(y_true, y_pred)
    normalizing_factor = np.abs(y_true.max() - y_pred.min())
    return _rmse / normalizing_factor

def mean_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean Error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.mean(y_true - y_pred)

def absolute_error(y_true: pd.Series, y_pred: pd.Series) -> np.array:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Mean_absolute_error
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Absolute error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.abs(y_true - y_pred)

def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Mean_absolute_error
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.mean(
        absolute_error(y_true, y_pred)
    )

def median_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Median absolute error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.median(
        absolute_error(y_true, y_pred)
    )

def variance_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Variance in absolute error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.var(
        absolute_error(y_true, y_pred)
    )

def iqr_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Interquartile Range in absolute error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return iqr(
        absolute_error(y_true, y_pred)
    )

def geometric_mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    formula comes from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Geometric Mean Absolute Percentage Error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return mstats.gmean(
        absolute_error(y_true, y_pred)
    )

def mean_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Percentage Error
    Formula taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean percentage error as a float.
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.mean(
        (y_true - y_pred) / y_true
    )

def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float: 
    """
    Mean Absolute Percentage Error
    Formula taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute percentage error as a float.
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    percentage_error = (y_true - y_pred) / y_true
    absolute_percentage_error = np.abs(percentage_error)
    return np.mean(absolute_percentage_error) * 100

def median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Median Absolute Percentage Error
    Formula taken from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Mean absolute percentage error as a float.
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    return np.median(np.abs(
        (y_true - y_pred) / y_true
    ))

def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Symmetric Mean Absolute Percentage Error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator /= 2
    return np.mean(numerator / denominator) * 100

def symmetric_median_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    formula comes from:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Symmetric Mean Absolute Percentage Error
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    denominator /= 2
    return np.median(numerator / denominator) * 100

def mean_arctangent_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error

    Note: result is NOT multiplied by 100
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    relative_error = (y_true - y_pred) / y_true
    abs_relative_error = np.abs(relative_error)
    return np.mean(np.arctan(
        abs_relative_error
    ))

def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Mean Absolute Scaled Error

    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    benchmark = math.floor(benchmark)
    # If no benchmark prediction provided - use naive forecasting
    if benchmark == 0:
        seasonality = 1
    else:
        seasonality = benchmark
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    error = y_true[seasonality:] - y_pred[seasonality:]
    naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]  
    naive_mae = mean_absolute_error(y_true[seasonality:], naive_forecast_error)
    mae = mean_absolute_error(y_true, y_pred)
    return mae / naive_mae

def normalized_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Normalized Absolute Error """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    error = y_true - y_pred
    square_difference = np.square(error - mae)
    summed_square_difference = np.sum(square_difference)
    num_observations = len(y_true) - 1
    return np.sqrt(summed_square_difference / num_observations)

def normalized_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Normalized Absolute Percentage Error """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    _mape = mean_absolute_percentage_error(y_true, y_pred)
    percentage_error = (y_true - y_pred) / y_true
    square_difference = np.square(percentage_error - _mape)
    summed_square_difference = np.sum(square_difference)
    num_observations = len(y_true) - 1 
    return np.sqrt(summed_square_difference / num_observations)

def root_mean_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Root Mean Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    percentage_error = (y_true - y_pred) / y_true
    square_percentage_error = np.square(percentage_error)
    return np.sqrt(np.mean(
        square_percentage_error
    ))

def root_median_squared_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Root Median Squared Percentage Error

    Note: result is NOT multiplied by 100
    """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    percentage_error = (y_true - y_pred) / y_true
    square_percentage_error = np.square(percentage_error)
    return np.sqrt(np.median(
        square_percentage_error
    ))

def root_mean_squared_scaled_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Root Mean Squared Scaled Error """
    benchmark = seasonal_decompose(y_true).seasonal.max()
    benchmark = math.floor(benchmark)
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    if benchmark == 0:
        seasonality = 1
    else:
        seasonality = benchmark
    
    error = y_true - y_pred
    naive_forecast_error = y_true[seasonality:] - y_true[:-seasonality]
    _mae = mean_absolute_error(y_true[seasonality:], naive_forecast_error)
    absolute_error_scaled = np.abs(error / _mae)
    squared_absolute_error_scaled = np.square(absolute_error_scaled)
    return np.sqrt(np.mean(
        squared_absolute_error_scaled
    ))

def integral_normalized_root_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Integral Normalized Root Squared Error """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    error = y_true - y_pred
    squared_error = np.square(error)
    normed_error = np.sqrt(np.sum(
        squared_error
    ))
    average_deviation = y_true - np.mean(y_true)
    squared_average_deviation = np.square(average_deviation)
    summed_squared_average_deviation = np.sum(
        squared_average_deviation
    )
    return normed_error / summed_squared_average_deviation 

def root_relative_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Root Relative Squared Error """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    error = y_true - y_pred
    squared_error = np.square(error)
    summed_squared_error = np.sum(squared_error)
    average_deviation = y_true - np.mean(y_true)
    squared_average_deviation = np.square(average_deviation)
    summed_squared_average_deviation = np.sum(
        squared_average_deviation
    )
    return np.sqrt(summed_squared_error / summed_squared_average_deviation)

def mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Mean Relative Error """
    return np.mean(_relative_error(y_true, y_pred))

def relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    absolute_error = np.abs(y_true - y_pred)
    summed_absolute_error = np.sum(absolute_error)
    average_deviation = y_true - np.mean(y_true)
    absolute_deviation = np.abs(average_deviation)
    summed_absolute_deviation = np.sum(
        absolute_deviation
    )
    return summed_absolute_error / summed_absolute_deviation

def mean_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Mean Relative Absolute Error """
    absolute_relative_error = np.abs(
        _relative_error(y_true, y_pred)
    )
    return np.mean(absolute_relative_error)

def median_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Median Relative Absolute Error """
    absolute_relative_error = np.abs(
        _relative_error(y_true, y_pred)
    )
    return np.median(absolute_relative_error)

def geometric_mean_relative_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    formula comes from:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * y_pred : pd.Series 
      The values of the forecast of the time series

    Note: all values are assumed to belong to the same time index

    Returns
    -------
    Geometric Mean Absolute Percentage Error
    """
    return mstats.gmean(np.abs(
        _relative_error(y_true, y_pred)
    ))
    
def mean_bounded_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(
        _bounded_relative_error(y_true, y_pred)
    )

def unscaled_mean_bounded_relative_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """ Unscaled Mean Bounded Relative Absolute Error """
    _mbrae = mean_bounded_relative_absolute_error(y_true, y_pred)
    return _mbrae / (1 - _mbrae)

def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """ Mean Directional Accuracy """
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    observations_increasing = np.sign(y_true[1:] - y_true[:-1])
    predictions_increasing = np.sign(y_pred[1:] - y_pred[:-1])
    directional_increase_count = (observations_increasing == predictions_increasing).astype(int)
    return np.mean(directional_increase_count)

def mahalanobis(x: float, distribution, covariance=None):
    """
    Compares a point to a distribution.  This allows you to compare
    a forecast over a distribution against the observed value.
    
    For more info check out:
    * https://github.com/EricSchles/datascience_book/blob/master/Applying%20Classification.ipynb
    * https://en.wikipedia.org/wiki/Mahalanobis_distance
    
    Parameters
    ----------
    * x : float
      The observation of the time series
    * distribution : arraylike
      The range of values for a forecast at a specific point.
    * covariance : arraylike
      The covariance matrix for the distribution
    
    Returns
    -------
    The mahalanobis distance between the point and distribution.
    """
    x_demeaned = x - np.mean(distribution)
    if covariance is None:
        covariance = np.cov(distribution.values.T)
    inverse_covariance = linalg.inv(covariance)
    left_term = np.dot(x_demeaned, inverse_covariance)
    mahalanobis_distance = np.dot(left_term, x_demeaned.T)
    return mahalanobis_distance.diagonal()

def mahalanobis_sum(y_true: pd.Series, y_pred: pd.DataFrame):
    """
    Compares points to a distributions.  This allows you to compare
    a forecast as distribution against the observed values.
    
    For more info check out:
    * https://github.com/EricSchles/datascience_book/blob/master/Applying%20Classification.ipynb
    * https://en.wikipedia.org/wiki/Mahalanobis_distance
    
    Parameters
    ----------
    * y_true : pd.Series
      The observations of the time series
    * distribution : pd.DataFrame
      The range of values for a forecast.
    
    Returns
    -------
    The summed mahalanobis distance between the observation and forecasts.
    """
    summation = 0
    for index in y_true.index:
        summation += mahalanobis(y_true[index], y_pred[index])
    return summation

def get_cdfs(y_true: pd.Series, y_pred: pd.Series):
    y_true = flatten_values(y_true)
    y_pred = flatten_values(y_pred)
    y_true_cdf = CDF.from_seq(y_true)
    y_pred_cdf = CDF.from_seq(y_pred)
    return y_true, y_pred

# statistic aliases
umbrae = unscaled_mean_bounded_relative_absolute_error
gmrae = geometric_mean_relative_absolute_error
mase = mean_absolute_scaled_error
smape = symmetric_mean_absolute_percentage_error
median_rae = median_relative_absolute_error
mean_rae = mean_relative_absolute_error
rmse = root_mean_squared_error
mape = mean_absolute_percentage_error
mbrae = mean_bounded_relative_absolute_error

def ad_fuller_test(timeseries: pd.Series):
    """
    Ad fuller documentation here:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa.stattools.adfuller
    
    Tests the unit root in a univariate process in the presence of serial
    correlation.
    
    Null hypothesis:
    there is a unit root
    
    Alternative hypothesis:
    there is no unit root, in otherwords the process is stationary.
    
    If the series has a unit root, then there is said to be no regression
    to the mean, while stationary processes will regress to the mean.
    """
    result = adfuller(timeseries)
    AdFullerResult = namedtuple('AdFullerResult', 'statistic pvalue')
    return AdFullerResult(result[0], result[1])

def kpss(timeseries: pd.Series, regression="c", nlags=None):
    """
    KPSS documentation here:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html#statsmodels.tsa.stattools.kpss
    
    Tests to see if the time series is trend stationary.
    
    Null hypothesis:
    The time series is level or trend stationary
    
    Alternative hypothesis:
    A unit root exists
    
    Parameters
    ----------
    * timeseries : pd.Series
      The observations of the time series
    * regression : str
      The time of model to use.
      If regression == 'c' check for stationarity around
      r_0.  Aka, difference against the mean
      If regression == 'ct' check for stationarity against
      the trend.  Aka, run OLS and run test against residuals.
    * nlags : str or int
      The number of lags to consider.
      If nlags == 'legacy' use:
      ceiling(12 * (n_observations/100)**(1/4)
      if nlags == 'auto' tune the number of lags
      if nlags == int, use that as number of lags.
    
    Note: all values are assumed to belong to the same time index

    Returns
    -------
    kpss_stat : float
        The KPSS test statistic.
    p_value : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Kwiatkowski et al. (1992), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).

    """
    result = kpss(
        timeseries, regression=regression,
        nlags=nlags, store=False
    )
    KPSSResult = namedtuple('KPSSResult', 'statistic pvalue')
    return KPSSResult(result[0], result[1])

def cointegration(y_zero: pd.Series, y_one: pd.Series, trend='c', maxlag=None, autolag='aic'):
    """
    Cointegration documentation:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html#statsmodels.tsa.stattools.coint
    
    Null hypothesis:
    No cointegration
    
    Assumption: y_zero and y_one are assumed to be integrated
    of order 1, I(1).
    
    Alternative hypothesis:
    cointegration exists between the two series.
    
    Definition: Cointegration - if two series are said to be 
    cointegrated then there exists a statistically significant connection between them.
    Specifically, some linear combination form a stationary time series.
    
    Implication: If y_zero, y_one both have a unit root, they are not stationary.
    But if some linear combination is stationary then there exists a long running
    relationship between the two series.
    
    Parameters
    ----------
    y_zero : pd.Series
        The first element in cointegrated system. Must be 1-d.
    y_one : pd.Series
        The remaining elements in cointegrated system.
    trend : str {'c', 'ct'}
        The trend term included in regression for cointegrating equation.

        * 'c' : constant.
        * 'ct' : constant and linear trend.
        * also available quadratic trend 'ctt', and no constant 'nc'.

    maxlag : None or int
        Argument for `adfuller`, largest or given number of lags.
    autolag : str
        Argument for `adfuller`, lag selection criterion.

        * If None, then maxlag lags are used without lag search.
        * If 'AIC' (default) or 'BIC', then the number of lags is chosen
          to minimize the corresponding information criterion.
        * 't-stat' based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
    """
    result = coint(y_zero, y_one)
    CointegrationResult = namedtuple('CointegrationResult', 'statistic pvalue')
    return CointegrationResult(result[0], result[1])

def chisquared_goodness_of_fit(y_true: pd.Series, y_pred: pd.Series, ddof=0, axis=0, lambda_=1):
    """
    The chi-squared goodness of fit test.
    
    Documentation:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.power_divergence.html
    
    Parameters
    ----------
    y_true : pd.Series
        Observed time series.
    y_pred : pd.Series
        Forecasted time series.
    ddof : int, optional
        "Delta degrees of freedom": adjustment to the degrees of freedom
        for the p-value.  The p-value is computed using a chi-squared
        distribution with ``k - 1 - ddof`` degrees of freedom, where `k`
        is the number of observed frequencies.  The default value of `ddof`
        is 0.
    axis : int or None, optional
        The axis of the broadcast result of `f_obs` and `f_exp` along which to
        apply the test.  If axis is None, all values in `f_obs` are treated
        as a single data set.  Default is 0.
    lambda_ : float or str, optional
        `lambda_` gives the power in the Cressie-Read power divergence
        statistic.  The default is 1.  For convenience, `lambda_` may be
        assigned one of the following strings, in which case the
        corresponding numerical value is used::
            String              Value   Description
            "pearson"             1     Pearson's chi-squared statistic.
                                        In this case, the function is
                                        equivalent to `stats.chisquare`.
            "log-likelihood"      0     Log-likelihood ratio. Also known as
                                        the G-test.
            "freeman-tukey"      -1/2   Freeman-Tukey statistic.
            "mod-log-likelihood" -1     Modified log-likelihood ratio.
            "neyman"             -2     Neyman's statistic.
            "cressie-read"        2/3   The power.
    Note: it is not recommended to change the lambda_ parameter unless
    there is very good reason, because this changes the test that is run.
    Please only change if you would like a different test.

    Returns
    -------
    stat : float or ndarray
        The Cressie-Read power divergence test statistic.  The value is
        a float if `axis` is None or if` `f_obs` and `f_exp` are 1-D.
    p : float or ndarray
        The p-value of the test.  The value is a float if `ddof` and the
        return value `stat` are scalars.
    """
    y_true = y_true[y_true > 0]
    y_pred = y_pred[y_true > 0]
    return stats.power_divergence(
        y_true, y_pred, ddof=ddof,
        axis=axis, lambda_=lambda_
    )

def het_white(y_true: pd.Series, resid: pd.Series):
    """
    White's Lagrange Multiplier Test for Heteroscedasticity.
    
    Null hypothesis:
    No heteroscedasticity for y_pred with respect to y_true.
    
    Note: This does not imply no serial correlation.
    
    Alternative hypothesis:
    heteroscedasticity exist for y_pred with respect to y_true.

    References: 
    * https://www.mathworks.com/help/econ/archtest.html
    * https://www.mathworks.com/help/econ/engles-arch-test.html

    Definition: Heteroscedasticity :=

    Heteroscedasticity means that the variance of a time series
    is not constant over time.  Therefore the variance over sliding
    window t,... t+i will differ from sliding window t+i+1,..t+j,
    where t is the initial time index, i, j are integers.
    
    Parameters
    ----------
    y_true : pd.Series
        observed values
    resid : pd.Series
        y_pred - y_true values
    nlags : int, default None
        Highest lag to use.
    ddof : int, default 0
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+q for an
        ARMA(p,q).

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    """
    result = diagnostic.het_white(
        resid, y_true
    )
    HetWhiteResult = namedtuple('HetWhiteResult', 'statistic pvalue')
    return HetWhiteResult(result[0], result[1])

def bds(timeseries, max_dim=10, epsilon=1.5, distance=None):
    """
    BDS documentation:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.bds.html#statsmodels.tsa.stattools.bds

    Generally speaking the Brock, Dechert, and Scheinkman test
    asks if the time series is a linear stochastic process.
    
    Null hypothesis:
    the time series is independent and identically distributed.
    
    Alternative hypothesis:
    The timeseries is not linear.
    
    For more information see:
    https://faculty.washington.edu/ezivot/econ584/notes/nonlinear.pdf
    
    Note: Another way to think of this test, is how chaotic is the series?
    Since you can tune epsilon or distance.  See:
    http://www.cs.bsu.edu/homepages/tliu/research/papers/jae1992v7ps25_liu.pdf
    for a discussion on this and a possible selection process for epsilon.
    
    Parameters
    ----------
    timeseries : pd.Series
        Observations of time series for which bds statistics is calculated.
    max_dim : int
        The maximum embedding dimension.  This number indicates the maximum
        for how far back in the series to go when looking for linearity.
    epsilon : float, optional
        Specifies the distance multiplier to use when computing the test
        statistic. 
        Note: epsilon is deviations are within epsilon standard deviations.
        So if epsilon is 1.5, we are checking to see if deviations, the 
        measure of correlation are within 1.5 standard deviations.
    distance : {float, None}, optional
        The threshold distance to use in calculating the correlation sum.
    
    Returns
    -------
    bds_stat : float
        The BDS statistic.
    pvalue : float
        The p-values associated with the BDS statistic.
    """
    result = bds(timeseries, max_dim=max_dim, epsilon=distance, distance=epsilon)
    BdsResult = namedtuple('BdsResult', 'statistic pvalue')
    return BdsResult(result[0], result[1])

def q_stat(timeseries):
    """
    autocorrelation function docs:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf

    Ljung-Box Q statistic docs:
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.q_stat.html#statsmodels.tsa.stattools.q_stat

    Tests whether any group of autocorrelations
    of a time series are different from zero.
    Specifically tests the overall randomness based
    on a number of lags.
    
    Null Hypothesis:
    The data are independently distributed
    
    Alternative Hypothesis:
    The data is not independently distributed,
    I.E. they exhibit serial correlation.
    
    Parameters
    ----------
    * timeseries : pd.Series
      The observations of the time series

    Returns
    -------
    Calculates the Ljung Box Q Statistics
    """
    autocorrelation_coefs = acf(timeseries)
    result = q_stat(autocorrelation_coefs)
    QstatResult = namedtuple('QstatResult', 'statistic pvalue')
    return QstatResult(result[0], result[1])

# percentage variance/ percentage daily growth within 2 days from actual forecast
# benchmarks against other models
# predict two weeks out - what did our predictions say two weeks out

def acorr_breusch_godfrey(resid, nlags=None):
    """
    Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.
    documentation can be found here:
    https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html

    This test looks for serial correlation in a timeseries.

    Definition: serial correlation :=
    Serial or auto correlation is a correlation of
    a signal with a delayed copy of itself.  
    
    The metric of correlation is the Pearson correlation
    and indicates a relationship with previous measurements
    in the series.  The presence of serial correlation can be used
    to understand periodicity.

    See this for more details:
    https://www.mathworks.com/help/signal/ug/find-periodicity-using-autocorrelation.html

    Null hypothesis:
    There is no serial correlation up to nlags.
    
    Alternative hypothesis:
    There is serial correlation.
    
    Parameters
    ----------
    resid : pd.Series
        Estimation results for which the residuals are tested for serial
        correlation.
    nlags : int, default None
        Number of lags to include in the auxiliary regression. (nlags is
        highest lag).
        if nlags is set to None then nlags is:
        ```
        nlags = np.trunc(12. * np.power(nobs / 100., 1 / 4.))
        nlags = int(nlags)
        ```

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    """
    result = diagnostic.acorr_breusch_godfrey(resid)
    AcorrBreuschGodfreyResult = namedtuple('BreuschGodfreyResult', 'statistic pvalue')
    return AcorrBreuschGodfreyResult(result[0], result[1])

def het_arch(resid: pd.Series, nlags=None, ddof=0):
    """
    Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH).
    
    Null hypothesis:
    No conditional heteroscedasticity
    
    Note: This does not imply no serial correlation.
    
    Alternative hypothesis:
    Conditional heteroscedasticity exists

    References: 
    * https://www.mathworks.com/help/econ/archtest.html
    * https://www.mathworks.com/help/econ/engles-arch-test.html

    Definition: Heteroscedasticity :=

    Heteroscedasticity means that the variance of a time series
    is not constant over time.  Therefore the variance over sliding
    window t,... t+i will differ from sliding window t+i+1,..t+j,
    where t is the initial time index, i, j are integers.
    
    Parameters
    ----------
    resid : pd.Series
        residuals from an estimation
    nlags : int, default None
        Highest lag to use.
    ddof : int, default 0
        If the residuals are from a regression, or ARMA estimation, then there
        are recommendations to correct the degrees of freedom by the number
        of parameters that have been estimated, for example ddof=p+q for an
        ARMA(p,q).

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic
    lmpval : float
        p-value for Lagrange multiplier test
    """
    result = diagnostic.het_arch(
        resid, nlags=nlags,
        autolag=None, ddof=ddof
    )
    HetArchResult = namedtuple('HetArchResult', 'statistic pvalue')
    return HetArchResult(result[0], result[1])

def breaks_cumsum(resid: pd.Series, ddof=0):
    """
    Cumulative summation test for parameter stability
    based on ols residuals.
    
    documentation:
    https://www.statsmodels.org/devel/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html#statsmodels.stats.diagnostic.breaks_cusumolsresid
    
    see:
    * https://en.wikipedia.org/wiki/Structural_break
    * https://www.stata.com/features/overview/cumulative-sum-test/
    Null Hypothesis:
    
    This test looks for 'breaks' or huge changes in the parameter of interest
    over time, to see if there is structural instability in the series.
    
    Parameters
    ----------
    resid : pd.Series
        An array of residuals.
    ddof : int
        The number of parameters in the OLS estimation, used as degrees
        of freedom correction for error variance.

    Returns
    -------
    sup_b : float
        The test statistic, maximum of absolute value of scaled cumulative OLS
        residuals.
    pval : float
        Probability of observing the data under the null hypothesis of no
        structural change, based on asymptotic distribution which is a Brownian
        Bridge
    """
    result = diagnostic.breaks_cusumolsresid(resid, ddof=ddof)
    BreaksCumSumResult = namedtuple('BreaksCumSumResult', 'statistic pvalue')
    return BreaksCumSumResult(result[0], result[1])

mean_absolute_deviation = mean_absolute_error 
