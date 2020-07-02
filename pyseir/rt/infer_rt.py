import math
import sys
from datetime import timedelta
import logging
import structlog

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy import signal
import us

from pyseir.load_data import load_county_metadata, get_all_fips_codes_for_a_state
from pyseir.utils import AggregationLevel, TimeseriesType, get_run_artifact_path, RunArtifact
from pyseir.rt.constants import InferRtConstants
from pyseir.rt import plotting, utils

rt_log = structlog.get_logger(__name__)


class RtInferenceEngine:
    """
    This class extends the analysis of Bettencourt et al to include mortality data in a
    pseudo-non-parametric inference of R_t.

    Parameters
    ----------
    fips: str
        State or County fips code
    include_testing_correction: bool
        If True, include a correction for testing increases and decreases.
    include_deaths: bool
        If True, include the deaths timeseries in the calculation. XCorrelated and Averaged
    """

    def __init__(
        self,
        fips,
        include_testing_correction=True,
        include_deaths=False,
        load_data_parent="pyseir",
    ):

        # Support injection of module that we use for loading data
        if "load_data" not in sys.modules:
            _temp = __import__(load_data_parent, globals(), locals(), ["load_data"], 0)
            self.load_data = _temp.load_data

        self.fips = fips
        self.include_testing_correction = include_testing_correction
        self.include_deaths = include_deaths

        # Load the InferRtConstants
        self.r_list = InferRtConstants.R_BUCKETS
        self.window_size = InferRtConstants.COUNT_SMOOTHING_WINDOW_SIZE
        self.kernel_std = InferRtConstants.COUNT_SMOOTHING_KERNEL_STD
        self.default_process_sigma = InferRtConstants.DEFAULT_PROCESS_SIGMA
        self.ref_date = InferRtConstants.REF_DATE
        self.confidence_intervals = InferRtConstants.CONFIDENCE_INTERVALS
        self.min_cases = InferRtConstants.MIN_COUNTS_TO_INFER
        self.min_deaths = InferRtConstants.MIN_COUNTS_TO_INFER
        self.min_ts_length = InferRtConstants.MIN_TIMESERIES_LENGTH
        self.serial_period = InferRtConstants.SERIAL_PERIOD
        self.max_scaling_sigma = InferRtConstants.MAX_SCALING_OF_SIGMA
        self.scale_sigma_from_count = InferRtConstants.SCALE_SIGMA_FROM_COUNT
        self.tail_suppression_correction = InferRtConstants.TAIL_SUPPRESSION_CORRECTION
        self.smooth_rt_map_composite = InferRtConstants.SMOOTH_RT_MAP_COMPOSITE
        self.rt_smoothing_window_size = InferRtConstants.RT_SMOOTHING_WINDOW_SIZE
        self.min_conf_width = InferRtConstants.MIN_CONF_WIDTH
        self.log = structlog.getLogger(Rt_Inference_Target="Test self.display_name")
        self.log_likelihood = None  # TODO: Add this later. Not in init.
        self.log.info(event="Running:")

        if len(fips) == 2:  # State FIPS are 2 digits
            self.agg_level = AggregationLevel.STATE
            self.state_obj = us.states.lookup(self.fips)
            self.state = self.state_obj.name

            (
                self.times,
                self.observed_new_cases,
                self.observed_new_deaths,
            ) = self.load_data.load_new_case_data_by_state(
                self.state,
                self.ref_date,
                include_testing_correction=self.include_testing_correction,
            )
            (
                self.times_raw_new_cases,
                self.raw_new_cases,
                _,
            ) = self.load_data.load_new_case_data_by_state(self.state, self.ref_date, False)

            self.display_name = self.state
        else:
            self.agg_level = AggregationLevel.COUNTY
            self.geo_metadata = load_county_metadata().set_index("fips").loc[fips].to_dict()
            self.state = self.geo_metadata["state"]
            self.state_obj = us.states.lookup(self.state)
            self.county = self.geo_metadata["county"]
            if self.county:
                self.display_name = self.county + ", " + self.state
            else:
                self.display_name = self.state

            (
                self.times,
                self.observed_new_cases,
                self.observed_new_deaths,
            ) = self.load_data.load_new_case_data_by_fips(
                self.fips,
                t0=self.ref_date,
                include_testing_correction=self.include_testing_correction,
            )
            (
                self.times_raw_new_cases,
                self.raw_new_cases,
                _,
            ) = self.load_data.load_new_case_data_by_state(self.state, self.ref_date, False)

        self.case_dates = [self.ref_date + timedelta(days=int(t)) for t in self.times]
        self.raw_new_case_dates = [
            self.ref_date + timedelta(days=int(t)) for t in self.times_raw_new_cases
        ]

    def get_timeseries(self, timeseries_type):
        """
        Given a timeseries type, return the dates, times, and requested values.

        Parameters
        ----------
        timeseries_type: TimeseriesType
            Which type of time-series to return.

        Returns
        -------
        dates: list(datetime)
            Dates for each observation
        times: list(int)
            Integer days since the reference date.
        timeseries:
            The requested timeseries.
        """
        timeseries_type = TimeseriesType(timeseries_type)

        if timeseries_type is TimeseriesType.NEW_CASES:
            return self.case_dates, self.times, self.observed_new_cases
        elif timeseries_type is TimeseriesType.RAW_NEW_CASES:
            return self.raw_new_case_dates, self.times_raw_new_cases, self.raw_new_cases
        elif timeseries_type is TimeseriesType.NEW_DEATHS or TimeseriesType.RAW_NEW_DEATHS:
            return self.case_dates, self.times, self.observed_new_deaths
        else:
            raise ValueError

    def evaluate_head_tail_suppression(self):
        """
        Evaluates how much time slows down (which suppresses Rt) as series approaches latest date
        """
        timeseries = pd.Series(1.0 * np.arange(0, 2 * self.window_size))
        smoothed = timeseries.rolling(
            self.window_size, win_type="gaussian", min_periods=self.kernel_std, center=True
        ).mean(std=self.kernel_std)
        delta = (smoothed - smoothed.shift()).tail(math.ceil(self.window_size / 2))

        return delta[delta < 1.0]

    def apply_gaussian_smoothing(self, timeseries_type, plot=True, smoothed_max_threshold=5):
        """
        Apply a rolling Gaussian window to smooth the data. This signature and
        returns match get_time_series, but will return a subset of the input
        time-series starting at the first non-zero value.

        Parameters
        ----------
        timeseries_type: TimeseriesType
            Which type of time-series to use.
        plot: bool
            If True, plot smoothed and original data.
        smoothed_max_threshold: int
            This parameter allows you to filter out entire series
            (e.g. NEW_DEATHS) when they do not contain high enough
            numeric values. This has been added to account for low-level
            constant smoothed values having a disproportionate effect on
            our final R(t) calculation, when all of their values are below
            this parameter.

        Returns
        -------
        dates: array-like
            Input data over a subset of indices available after windowing.
        times: array-like
            Output integers since the reference date.
        smoothed: array-like
            Gaussian smoothed data.


        """
        timeseries_type = TimeseriesType(timeseries_type)
        dates, times, timeseries = self.get_timeseries(timeseries_type)
        self.log = self.log.bind(timeseries_type=timeseries_type.value)

        # Don't even try if the timeseries is too short.
        # TODO: This referenced a quirk of hospitalizations. So may be stale as of 1 July 2020.
        if len(timeseries) < self.min_ts_length:
            return [], [], []

        # Remove Outliers Before Smoothing. Replaces a value if the current is more than 10 std
        # from the 14 day trailing mean and std
        timeseries = utils.replace_outliers(pd.Series(timeseries), log=self.log)

        # Smoothing no longer involves rounding
        smoothed = timeseries.rolling(
            self.window_size, win_type="gaussian", min_periods=self.kernel_std, center=True
        ).mean(std=self.kernel_std)

        # Retain logic for detecting what would be nonzero values if rounded
        nonzeros = [idx for idx, val in enumerate(smoothed.round()) if val != 0]

        if smoothed.empty:
            idx_start = 0
        elif max(smoothed) < smoothed_max_threshold:
            # skip the entire array.
            idx_start = len(smoothed)
        else:
            idx_start = nonzeros[0]

        smoothed = smoothed.iloc[idx_start:]

        # Only plot counts and smoothed timeseries for cases
        if plot and timeseries_type == TimeseriesType.NEW_CASES and len(smoothed) > 0:
            fig = plotting.plot_smoothing(
                x=dates,
                original=timeseries.loc[smoothed.index],
                processed=smoothed,
                timeseries_type=timeseries_type,
            )
            output_path = get_run_artifact_path(self.fips, RunArtifact.RT_SMOOTHING_REPORT)
            fig.savefig(output_path, bbox_inches="tight")

        return dates, times, smoothed

    def highest_density_interval(self, posteriors, ci):
        """
        Given a PMF, generate the confidence bands.

        Parameters
        ----------
        posteriors: pd.DataFrame
            Probability Mass Function to compute intervals for.
        ci: float
            Float confidence interval. Value of 0.95 will compute the upper and
            lower bounds.

        Returns
        -------
        ci_low: np.array
            Low confidence intervals.
        ci_high: np.array
            High confidence intervals.
        """
        posterior_cdfs = posteriors.values.cumsum(axis=0)
        low_idx_list = np.argmin(np.abs(posterior_cdfs - (1 - ci)), axis=0)
        high_idx_list = np.argmin(np.abs(posterior_cdfs - ci), axis=0)
        ci_low = self.r_list[low_idx_list]
        ci_high = self.r_list[high_idx_list]
        return ci_low, ci_high

    def make_process_matrix(self, timeseries_scale=InferRtConstants.SCALE_SIGMA_FROM_COUNT):
        """ Externalizes process of generating the Gaussian process matrix adding the following:
        1) Auto adjusts sigma from its default value for low counts - scales sigma up as
           1/sqrt(count) up to a maximum factor of MAX_SCALING_OF_SIGMA
        2) Ensures the smoothing (of the posterior when creating the prior) is symmetric
           in R so that this process does not move argmax (the peak in probability)
        """
        # TODO FOR ALEX: Please expand this and describe more clearly these cutoffs
        use_sigma = (
            min(
                self.max_scaling_sigma,
                max(1.0, math.sqrt(self.scale_sigma_from_count / timeseries_scale)),
            )
            * self.default_process_sigma
        )

        process_matrix = sps.norm(loc=self.r_list, scale=use_sigma).pdf(self.r_list[:, None])

        # process_matrix applies gaussian smoothing to the previous posterior to make the prior.
        # But when the gaussian is wide much of its distribution function can be outside of the
        # range Reff = (0,10). When this happens the smoothing is not symmetric in R space. For
        # R<1, when posteriors[previous_day]).argmax() < 50, this asymmetry can push the argmax of
        # the prior >10 Reff bins (delta R = .2) on each new day. This was a large systematic error.

        # Ensure smoothing window is symmetric in X direction around diagonal
        # to avoid systematic drift towards middle (Reff = 5). This is done by
        # ensuring the following matrix values are 0:
        # 1 0 0 0 0 0 ... 0 0 0 0 0 0
        # * * * 0 0 0 ... 0 0 0 0 0 0
        # ...
        # * * * * * * ... * * * * 0 0
        # * * * * * * ... * * * * * *
        # 0 0 * * * * ... * * * * * *
        # ...
        # 0 0 0 0 0 0 ... 0 0 0 * * *
        # 0 0 0 0 0 0 ... 0 0 0 0 0 1
        sz = len(self.r_list)
        for row in range(0, sz):
            if row < (sz - 1) / 2:
                process_matrix[row, 2 * row + 1 : sz] = 0.0
            elif row > (sz - 1) / 2:
                process_matrix[row, 0 : sz - 2 * (sz - row)] = 0.0

        # (3a) Normalize all rows to sum to 1
        row_sums = process_matrix.sum(axis=1)
        for row in range(0, sz):
            process_matrix[row] = process_matrix[row] / row_sums[row]

        return use_sigma, process_matrix

    def get_posteriors(self, timeseries_type, plot=False):
        """
        Generate posteriors for R_t.

        Parameters
        ----------
        ----------
        timeseries_type: TimeseriesType
            New X per day (cases, deaths etc).
        plot: bool
            If True, plot a cool looking est of posteriors.

        Returns
        -------
        dates: array-like
            Input data over a subset of indices available after windowing.
        times: array-like
            Output integers since the reference date.
        posteriors: pd.DataFrame
            Posterior estimates for each timestamp with non-zero data.
        start_idx: int
            Index of first Rt value calculated from input data series
            #TODO figure out why this value sometimes truncates the series
            
        """
        # Propagate self.min_[cases,deaths] into apply_gaussian_smoothing where used to abort
        # processing of timeseries without high enough counts
        smoothed_max_threshold = (
            self.min_cases if TimeseriesType.NEW_CASES == timeseries_type else self.min_deaths
        )
        dates, times, timeseries = self.apply_gaussian_smoothing(
            timeseries_type, smoothed_max_threshold=smoothed_max_threshold
        )

        if len(timeseries) == 0:
            rt_log.info(
                "%s: empty timeseries %s, skipping" % (self.display_name, timeseries_type.value)
            )
            return None, None, None, None
        else:
            rt_log.info(
                "%s: Analyzing posteriors for timeseries %s"
                % (self.display_name, timeseries_type.value)
            )

        # (1) Calculate Lambda (the Poisson likelihood given the data) based on
        # the observed increase from t-1 cases to t cases.
        lam = timeseries[:-1].values * np.exp((self.r_list[:, None] - 1) / self.serial_period)

        # (2) Calculate each day's likelihood over R_t
        # Originally smoothed counts were rounded (as needed for sps.poisson.pmf below) which
        # doesn't work well for low counts and introduces artifacts at rounding transitions. Now
        # calculate for both ceiling and floor values and interpolate between to get smooth
        # behaviour
        ts_floor = timeseries.apply(np.floor).astype(int)
        ts_ceil = timeseries.apply(np.ceil).astype(int)
        ts_frac = timeseries - ts_floor

        likelihoods_floor = pd.DataFrame(
            data=sps.poisson.pmf(ts_floor[1:].values, lam),
            index=self.r_list,
            columns=timeseries.index[1:],
        )
        likelihoods_ceil = pd.DataFrame(
            data=sps.poisson.pmf(ts_ceil[1:].values, lam),
            index=self.r_list,
            columns=timeseries.index[1:],
        )
        # Interpolate between value for ceiling and floor of smoothed counts
        likelihoods = ts_frac * likelihoods_ceil + (1 - ts_frac) * likelihoods_floor

        # (3) Create the (now scaled up for low counts) Gaussian Matrix
        (current_sigma, process_matrix) = self.make_process_matrix(timeseries.median())

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)

        # (4) Calculate the initial prior. Gamma mean of "a" with mode of "a-1".
        prior0 = sps.gamma(a=2.5).pdf(self.r_list)
        prior0 /= prior0.sum()

        reinit_prior = sps.gamma(a=2).pdf(self.r_list)
        reinit_prior /= reinit_prior.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=self.r_list, columns=timeseries.index, data={timeseries.index[0]: prior0}
        )

        # We said we'd keep track of the sum of the log of the probability
        # of the data for maximum likelihood calculation.
        log_likelihood = 0.0

        # Initialize timeseries scale (used for auto sigma)
        scale = timeseries.head(1).item()

        # Setup monitoring for Reff lagging signal in daily likelihood
        monitor = utils.LagMonitor(debug=False)  # Set debug=True for detailed printout of daily lag

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(timeseries.index[:-1], timeseries.index[1:]):

            # Keep track of exponential moving average of scale of counts of timeseries
            scale = 0.9 * scale + 0.1 * timeseries[current_day]

            # Calculate process matrix for each day
            (current_sigma, process_matrix) = self.make_process_matrix(scale)

            # (5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]

            # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior

            # (5c) Calculate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            if denominator == 0:
                # Restart the baysian learning for the remaining series.
                # This is necessary since otherwise NaN values
                # will be inferred for all future days, after seeing
                # a single (smoothed) zero value.
                #
                # We understand that restarting the posteriors with the
                # re-initial prior may incur a start-up artifact as the posterior
                # restabilizes, but we believe it's the current best
                # solution for municipalities that have smoothed cases and
                # deaths that dip down to zero, but then start to increase
                # again.

                posteriors[current_day] = reinit_prior
            else:
                posteriors[current_day] = numerator / denominator

            # Monitors if posterior is lagging excessively behind signal in likelihood
            # TODO future can return cumulative lag and use to scale sigma up only when needed
            monitor.evaluate_lag_using_argmaxes(
                current_day,
                current_sigma,
                posteriors[previous_day].argmax(),
                current_prior.argmax(),
                likelihoods[current_day].argmax(),
                numerator.argmax(),
            )

            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)

        self.log_likelihood = log_likelihood

        if plot:
            plotting.plot_posteriors(x=posteriors)  # Returns Figure.
            # The interpreter will handle this as it sees fit. Normal builds never call plot flag.

        start_idx = -len(posteriors.columns)

        return dates[start_idx:], times[start_idx:], posteriors, start_idx

    def get_available_timeseries(self):
        """
        Determine available timeseries for Rt inference calculation 
        with constraints below

        Returns
        -------
        available_timeseries: 
          array of available timeseries saved as TimeseriesType
        """
        available_timeseries = []
        IDX_OF_COUNTS = 2
        cases = self.get_timeseries(TimeseriesType.NEW_CASES.value)[IDX_OF_COUNTS]
        deaths = self.get_timeseries(TimeseriesType.NEW_DEATHS.value)[IDX_OF_COUNTS]

        if np.sum(cases) > self.min_cases:
            available_timeseries.append(TimeseriesType.NEW_CASES)
            available_timeseries.append(TimeseriesType.RAW_NEW_CASES)

        if np.sum(deaths) > self.min_deaths:
            available_timeseries.append(TimeseriesType.RAW_NEW_DEATHS)
            available_timeseries.append(TimeseriesType.NEW_DEATHS)

        return available_timeseries

    def infer_all(self, plot=True, shift_deaths=0):
        """
        Infer R_t from all available data sources.

        Parameters
        ----------
        plot: bool
            If True, generate a plot of the inference.
        shift_deaths: int
            Shift the death time series by this amount with respect to cases
            (when plotting only, does not shift the returned result).

        Returns
        -------
        inference_results: pd.DataFrame
            Columns containing MAP estimates and confidence intervals.
        """
        df_all = None
        available_timeseries = self.get_available_timeseries()

        for timeseries_type in available_timeseries:
            # Add Raw Data Output to Output DataFrame
            dates_raw, times_raw, timeseries_raw = self.get_timeseries(timeseries_type)
            df_raw = pd.DataFrame()
            df_raw["date"] = dates_raw
            df_raw = df_raw.set_index("date")
            df_raw[f"{timeseries_type.value}"] = timeseries_raw

            df = pd.DataFrame()
            dates, times, posteriors, start_idx = self.get_posteriors(timeseries_type)
            # Note that it is possible for the dates to be missing days
            # This can cause problems when:
            #   1) computing posteriors that assume continuous data (above),
            #   2) when merging data with variable keys
            if posteriors is None:
                continue

            df[f"Rt_MAP__{timeseries_type.value}"] = posteriors.idxmax()
            for ci in self.confidence_intervals:
                ci_low, ci_high = self.highest_density_interval(posteriors, ci=ci)

                low_val = 1 - ci
                high_val = ci
                df[f"Rt_ci{int(math.floor(100 * low_val))}__{timeseries_type.value}"] = ci_low
                df[f"Rt_ci{int(math.floor(100 * high_val))}__{timeseries_type.value}"] = ci_high

            df["date"] = dates
            df = df.set_index("date")

            if df_all is None:
                df_all = df
            else:
                # To avoid any surprises merging the data, keep only the keys from the case data
                # which will be the first added to df_all. So merge with how ="left" rather than
                # "outer"
                df_all = df_all.merge(df_raw, left_index=True, right_index=True, how="left")
                df_all = df_all.merge(df, left_index=True, right_index=True, how="left")

            # ------------------------------------------------
            # Compute the indicator lag using the curvature
            # alignment method.
            # ------------------------------------------------
            if (
                timeseries_type in (TimeseriesType.NEW_DEATHS,)
                and f"Rt_MAP__{TimeseriesType.NEW_CASES.value}" in df_all.columns
            ):

                # Go back up to 30 days or the max time series length we have if shorter.
                last_idx = max(-21, -len(df))
                series_a = df_all[f"Rt_MAP__{TimeseriesType.NEW_CASES.value}"].iloc[-last_idx:]
                series_b = df_all[f"Rt_MAP__{timeseries_type.value}"].iloc[-last_idx:]

                shift_in_days = self.align_time_series(series_a=series_a, series_b=series_b,)

                df_all[f"lag_days__{timeseries_type.value}"] = shift_in_days
                logging.debug(
                    "Using timeshift of: %s for timeseries type: %s ",
                    shift_in_days,
                    timeseries_type,
                )
                # Shift all the columns.
                for col in df_all.columns:
                    if timeseries_type.value in col:
                        df_all[col] = df_all[col].shift(shift_in_days)
                        # Extend death and hopitalization rt signals beyond
                        # shift to avoid sudden jumps in composite metric.
                        #
                        # N.B interpolate() behaves differently depending on the location
                        # of the missing values: For any nans appearing in between valid
                        # elements of the series, an interpolated value is filled in.
                        # For values at the end of the series, the last *valid* value is used.
                        logging.debug("Filling in %s missing values", shift_in_days)
                        df_all[col] = df_all[col].interpolate(
                            limit_direction="forward", method="linear"
                        )

        if df_all is None:
            logging.warning("Inference not possible for fips: %s", self.fips)
            return None

        if self.include_deaths and "Rt_MAP__new_deaths" in df_all and "Rt_MAP__new_cases" in df_all:
            df_all["Rt_MAP_composite"] = np.nanmean(
                df_all[["Rt_MAP__new_cases", "Rt_MAP__new_deaths"]], axis=1
            )
            # Just use the Stdev of cases. A correlated quadrature summed error
            # would be better, but is also more confusing and difficult to fix
            # discontinuities between death and case errors since deaths are
            # only available for a subset. Systematic errors are much larger in
            # any case.
            df_all["Rt_ci95_composite"] = df_all["Rt_ci95__new_cases"]

        elif "Rt_MAP__new_cases" in df_all:
            df_all["Rt_MAP_composite"] = df_all["Rt_MAP__new_cases"]
            df_all["Rt_ci95_composite"] = df_all["Rt_ci95__new_cases"]

        # Correct for tail suppression
        suppression = 1.0 * np.ones(len(df_all))
        if self.tail_suppression_correction > 0.0:
            tail_sup = self.evaluate_head_tail_suppression()
            # Calculate rt suppression by smoothing delay at tail of sequence
            suppression = np.concatenate(
                [1.0 * np.ones(len(df_all) - len(tail_sup)), tail_sup.values]
            )
            # Adjust rt by undoing the suppression
            df_all["Rt_MAP_composite"] = (df_all["Rt_MAP_composite"] - 1.0) / np.power(
                suppression, self.tail_suppression_correction
            ) + 1.0

        # Optionally Smooth just Rt_MAP_composite.
        # Note this doesn't lag in time and preserves integral of Rteff over time
        for i in range(0, self.smooth_rt_map_composite):
            kernel_width = round(self.rt_smoothing_window_size / 4)
            smoothed = (
                df_all["Rt_MAP_composite"]
                .rolling(
                    self.rt_smoothing_window_size,
                    win_type="gaussian",
                    min_periods=kernel_width,
                    center=True,
                )
                .mean(std=kernel_width)
            )

            # Adjust down confidence interval due to count smoothing over kernel_width values but
            # not below .2
            df_all["Rt_MAP_composite"] = smoothed
            df_all["Rt_ci95_composite"] = (
                (df_all["Rt_ci95_composite"] - df_all["Rt_MAP_composite"])
                / math.sqrt(
                    2.0 * kernel_width  # averaging over many points reduces confidence interval
                )
                / np.power(suppression, self.tail_suppression_correction / 2)
            ).apply(lambda v: max(v, self.min_conf_width)) + df_all["Rt_MAP_composite"]

        if plot:
            fig = plotting.plot_rt(
                df=df_all,
                include_deaths=self.include_deaths,
                shift_deaths=shift_deaths,
                display_name=self.display_name,
            )
            output_path = get_run_artifact_path(self.fips, RunArtifact.RT_INFERENCE_REPORT)
            fig.savefig(output_path, bbox_inches="tight")
        if df_all.empty:
            logging.warning("Inference not possible for fips: %s", self.fips)
        return df_all

    @staticmethod
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

    @staticmethod
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
        shifts = range(-21, 5)
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

    @classmethod
    def run_for_fips(cls, fips):
        try:
            engine = cls(fips)
            return engine.infer_all()
        except Exception:
            logging.exception("run_for_fips failed")
            return None


def run_state(state, states_only=False):
    """
    Run the R_t inference for each county in a state.

    Parameters
    ----------
    state: str
        State to run against.
    states_only: bool
        If True only run the state level.
    """
    state_obj = us.states.lookup(state)
    df = RtInferenceEngine.run_for_fips(state_obj.fips)
    output_path = get_run_artifact_path(state_obj.fips, RunArtifact.RT_INFERENCE_RESULT)
    if df is None or df.empty:
        logging.error("Empty DataFrame encountered! No RtInference results available for %s", state)
    else:
        df.to_json(output_path)

    # Run the counties.
    if not states_only:
        all_fips = get_all_fips_codes_for_a_state(state)

        # Something in here doesn't like multiprocessing...
        rt_inferences = all_fips.map(lambda x: RtInferenceEngine.run_for_fips(x)).tolist()

        for fips, rt_inference in zip(all_fips, rt_inferences):
            county_output_file = get_run_artifact_path(fips, RunArtifact.RT_INFERENCE_RESULT)
            if rt_inference is not None:
                rt_inference.to_json(county_output_file)


def run_county(fips):
    """
    Run the R_t inference for each county in a state.

    Parameters
    ----------
    fips: str
        County fips to run against
    """
    if not fips:
        return None

    df = RtInferenceEngine.run_for_fips(fips)
    county_output_file = get_run_artifact_path(fips, RunArtifact.RT_INFERENCE_RESULT)
    if df is not None and not df.empty:
        df.to_json(county_output_file)
