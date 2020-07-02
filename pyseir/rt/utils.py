import logging

import numpy as np

from pyseir.rt.constants import InferRtConstants

utils_log = logging.getLogger(__name__)


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


def replace_outliers(
    x,
    log,
    local_lookback_window=InferRtConstants.LOCAL_LOOKBACK_WINDOW,
    z_threshold=InferRtConstants.Z_THRESHOLD,
    min_mean_to_consider=InferRtConstants.MIN_MEAN_TO_CONSIDER,
):
    """
    Take a pandas.Series, apply an outlier filter, and return a pandas.Series.

    This outlier detector looks at the z score of the current value compared to the mean and std
    derived from the previous N samples, where N is the local_lookback_window.

    For points where the z score is greater than z_threshold, a check is made to make sure the mean
    of the last N samples is at least min_mean_to_consider. This makes sure we don't filter on the
    initial case where values go from all zeros to a one. If that threshold is met, the value is
    then replaced with the linear interpolation between the two nearest neighbors.


    Parameters
    ----------
    x
        Input pandas.Series with the values to analyze
    log
        Logger instance
    local_lookback_window
        The length of the rolling window to look back and calculate the mean and std to baseline the
        z score. NB: We require the window to be full before returning any result.
    z_threshold
        The minimum z score needed to trigger the replacement
    min_mean_to_consider
        Threshold to skip low n cases, especially the degenerate case where a long list of zeros
        becomes a 1. This requires that the rolling mean of previous values must be greater than
        or equal to min_mean_to_consider to be replaced.
    Returns
    -------
    x
        pandas.Series with any triggered outliers replaced
    """
    # Small epsilon to prevent divide by 0 errors.
    EPSILON = 1e-8

    # Calculate Z Score
    r = x.rolling(window=local_lookback_window, min_periods=local_lookback_window, center=False)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z_score = (x - m) / (s + EPSILON)
    possible_changes_idx = np.flatnonzero(z_score > z_threshold)
    changed_idx = []
    changed_value = []
    changed_snippets = []

    for idx in possible_changes_idx:
        if m[idx] > min_mean_to_consider:
            changed_idx.append(idx)
            changed_value.append(int(x[idx]))
            slicer = slice(idx - local_lookback_window, idx + local_lookback_window)
            changed_snippets.append(x[slicer].astype(int).tolist())
            try:
                x[idx] = np.mean([x.iloc[idx - 1], x.iloc[idx + 1]])
            except IndexError:  # Value to replace can be newest and fail on x[idx+1].
                # If so, just use previous.
                x[idx] = x[idx - 1]

    if len(changed_idx) > 0:
        log.info(
            event="Replacing Outliers:",
            outlier_values=changed_value,
            z_score=z_score[changed_idx].astype(int).tolist(),
            where=changed_idx,
            snippets=changed_snippets,
        )

    return x
