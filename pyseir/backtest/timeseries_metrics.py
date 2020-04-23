import pandas as pd
from statsmodels.tsa.api import (
    adfuller, bds, coint, kpss, acf, q_stat
)
from statsmodels.stats import diagnostic
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import namedtuple
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
    error_type = ErrorType(error_type)
    if error_type is ErrorType.RMSE:
        return 'root of mean squared error'
    if error_type is ErrorType.NRMSE:
        return 'normalized root of mean squared error'
    if error_type is ErrorType.ERROR:
        return 'residual'
    if error_type is ErrorType.RELATIVE_ERROR:
        return 'percentage error'
    if error_type is ErrorType.PERCENTAGE_ABS_ERROR:
        return 'Absolute percentage error'
    if error_type is ErrorType.SYMMETRIC_ABS_ERROR:
        return 'Absolute error normalized by sum of prediction and observation'
    if error_type is ErrorType.ABS_ERROR:
        return 'Absolate error'


class TimeSeriesMetrics:

    def __init__(self):
        self.error = None

    def flatten_values(self, series):
        """
        Flatten timeseries into array.
        """
        if isinstance(series, pd.Series) or isinstance(series, pd.DataFrame):
            return series.values.ravel()
        return series


    def calculate_error(self, y_true, y_pred, error_type, missing_observation=None):
        """

        """
        y_true = self.flatten_values(y_true)
        y_pred = self.flatten_values(y_pred)

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
            error = np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())
        elif error_type is ErrorType.NRMSE:
            error = np.sqrt(np.square(np.subtract(y_true, y_pred)).mean()) / y_true.mean()

        if error_type not in [ErrorType.RMSE, ErrorType.NRMSE]:
            error[invalid_observations] = np.nan

        return error


    def calculate_error_measure(self, error, measure):
        """

        """

        measure = Measure(measure)

        if measure is Measure.MEAN:
            return np.mean(error)
        elif measure is Measure.MEDIAN:
            return np.median(error)
        elif measure is Measure.VAR:
            return np.var(error)
        elif measure is Measure.VAR:
            return np.std(error)
        elif measure is Measure.CI_95:
            return (np.quantile(error, (0.025, 0.975)))
        elif measure is Measure.CI_68:
            return (np.quantile(error, (0.16, 0.84)))

    def ad_fuller_test(self, timeseries: pd.Series):
        """
        Ad fuller documentation here:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html#statsmodels.tsa
        .stattools.adfuller

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

    def kpss(self, timeseries: pd.Series, regression="c", nlags=None):
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

    def cointegration(self, y_zero: pd.Series, y_one: pd.Series, trend='c', maxlag=None, autolag='aic'):
        """
        Cointegration documentation:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html#statsmodels.tsa.stattools
        .coint

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

    def chisquared_goodness_of_fit(self, y_true: pd.Series, y_pred: pd.Series, ddof=0, axis=0, lambda_=1):
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

    def bds(self, timeseries, max_dim=10, epsilon=1.5, distance=None):
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

    def q_stat(self, timeseries):
        """
        autocorrelation function docs:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html#statsmodels.tsa.stattools.acf

        Ljung-Box Q statistic docs:
        https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.q_stat.html#statsmodels.tsa.stattools
        .q_stat

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

    def acorr_breusch_godfrey(self, resid, nlags=None):
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

    def het_arch(self, resid: pd.Series, nlags=None, ddof=0):
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

    def breaks_cumsum(welf, resid: pd.Series, ddof=0):
        """
        Cumulative summation test for parameter stability
        based on ols residuals.

        documentation:
        https://www.statsmodels.org/devel/generated/statsmodels.stats.diagnostic.breaks_cusumolsresid.html
        #statsmodels.stats.diagnostic.breaks_cusumolsresid

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
