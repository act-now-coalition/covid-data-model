import logging
import numpy as np
from scipy import signal

from pyseir.rt.constants import InferRtConstants

utils_log = logging.getLogger(__name__)


# PR598 Request by Greater New Orlean Public Health to have a consistent Rt across the following:
NEW_ORLEANS_FIPS = (
    "22051",  # Jefferson
    "22071",  # Orleans
    "22075",  # Plaquemines
    "22087",  # St Bernard
    "22103",  # St Tammany
)


class LagMonitor:
    """
    Monitors lag in posterior relative to driving likelihood as Bayesian update is repeatedly
    applied. If lag is severe logs a short warning (or a longer one if debug is enabled).
    """

    # TODO add method to support issuing any warning at the end of processing

    def __init__(self, r_bin_threshold=5, lag_fraction=0.8, days_threshold=4, debug=False):
        self.lag_fraction = lag_fraction
        self.days_threshold = days_threshold
        self.r_bin_threshold = r_bin_threshold
        self.debug = debug
        self.last_lag = 0
        self._reset_()

    def _reset_(self):
        self.lag_days_running = list()
        self.max_lag = 0
        self.total_lag = 0

    def evaluate_lag_using_argmaxes(
        self, current_day, current_sigma, prev_post_am, prior_am, like_am, post_am
    ):
        # Test if there is lag by checking whether pull of consistent likelihood in one direction
        # can move the value fast enough (as determined by sigma). Looking at argmax values of
        # previous posterior/current prior, current likelihood and current posterior
        p_po_am = prev_post_am
        c_pr_am = prior_am
        c_li_am = like_am
        c_po_am = post_am
        driving_likelihood = c_li_am - c_pr_am
        lag_after_update = c_li_am - c_po_am

        # Is this day lagging more than threshold applied to drive?
        compare_lag = round(self.lag_fraction * abs(driving_likelihood))
        noLag = (
            current_day < 12  # needs to settle in
            or abs(lag_after_update) < self.r_bin_threshold  # *.02 for threshold in Reff
            or abs(lag_after_update)
            < compare_lag  # Able to move 1/3 of drive per day -> 3 days lag
            or self.last_lag * lag_after_update < 0  # Drive switched directions
        )
        if self.debug:
            ind = "ok" if noLag else "LAGGING"
            print(
                (
                    "day {d} {ind}... prior = {pr}, likelihood drive {dd} -> update {up} (remaining "
                    "lag = {lag} vs {cmp}) yielding posterior = {po}"
                ).format(
                    d=current_day,
                    pr=p_po_am,
                    dd=driving_likelihood,
                    up=c_po_am - c_pr_am,
                    po=c_po_am,
                    lag=lag_after_update,
                    cmp=compare_lag,
                    ind=ind,
                )
            )
        if noLag:  # End of lagging sequence of days
            if len(self.lag_days_running) >= self.days_threshold:  # Need 3 days running to warn
                length = len(self.lag_days_running)
                utils_log.info(
                    (
                        "Reff lagged likelihood (max = %.2f, mean = %.2f) with sigma %.3f for %d days"
                        "(from %d to %d)"
                    )
                    % (
                        0.02 * self.max_lag,
                        0.02 * self.total_lag / length,
                        current_sigma,
                        length,
                        current_day - length,
                        current_day,
                    )
                )
            self._reset_()
        if abs(lag_after_update) >= 4:  # Start tracking any new lag
            self.lag_days_running.append(
                [
                    current_day,
                    p_po_am,
                    driving_likelihood,
                    c_po_am - c_pr_am,
                    c_po_am,
                    lag_after_update,
                ]
            )
            self.total_lag += lag_after_update
            if abs(lag_after_update) > abs(self.max_lag):
                self.max_lag = lag_after_update
        self.last_lag = lag_after_update


def ewma_smoothing(series, tau=5):
    """
    Exponentially weighted moving average of a series.

    Parameters
    ----------
    series: array-like
        Series to convolve.
    tau: float
        Decay factor.

    Returns
    -------
    smoothed: array-like
        Smoothed series.
    """
    exp_window = signal.exponential(2 * tau, 0, tau, False)[::-1]
    exp_window /= exp_window.sum()
    smoothed = signal.convolve(series, exp_window, mode="same")
    return smoothed


def align_time_series(series_a, series_b):
    """
    Identify the optimal time shift between two data series based on
    maximal cross-correlation of their derivatives.

    Parameters
    ----------
    series_a: pd.Series
        Reference series to cross-correlate against.
    series_b: pd.Series
        Reference series to shift and cross-correlate against.

    Returns
    -------
    shift: int
        A shift period applied to series b that aligns to series a
    """
    shifts = InferRtConstants.XCOR_DAY_RANGE
    valid_shifts = []
    xcor = []
    np.random.seed(InferRtConstants.RNG_SEED)  # Xcor has some stochastic FFT elements.
    _series_a = np.diff(series_a)

    for i in shifts:
        series_b_shifted = np.diff(series_b.shift(i))
        valid = ~np.isnan(_series_a) & ~np.isnan(series_b_shifted)
        if len(series_b_shifted[valid]) > 0:
            xcor.append(signal.correlate(_series_a[valid], series_b_shifted[valid]).mean())
            valid_shifts.append(i)
    if len(valid_shifts) > 0:
        return valid_shifts[np.argmax(xcor)]
    else:
        return 0
