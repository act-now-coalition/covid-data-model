import math
from datetime import timedelta
import logging
import structlog

import numpy as np
import pandas as pd
from scipy import stats as sps
from matplotlib import pyplot as plt

from pyseir import load_data
from pyseir.utils import TimeseriesType, get_run_artifact_path, RunArtifact
from pyseir.rt.constants import InferRtConstants
from pyseir.rt import plotting, utils

rt_log = structlog.get_logger(__name__)


def run_rt_for_fips(fips, figure_collector=None):
    """Entry Point for Infer Rt"""

    # Generate the Data Packet to Pass to RtInferenceEngine
    input_df = _generate_input_data(
        fips, include_testing_correction=True, figure_collector=figure_collector
    )

    # Save a reference to instantiated engine (eventually I want to pull out the figure generation
    # and saving so that I don't have to pass a display_name and fips into the class
    engine = RtInferenceEngine(
        data=input_df, display_name=_get_display_name(fips), fips=fips, include_deaths=False
    )

    # Generate the output DataFrame (consider renaming the function infer_all to make it clearer)
    output_df = engine.infer_all()

    # Save the output to json for downstream repacking and incorporation.
    county_output_file = get_run_artifact_path(fips, RunArtifact.RT_INFERENCE_RESULT)
    if output_df is not None and not output_df.empty:
        output_df.to_json(county_output_file)


def _get_display_name(fips) -> str:
    """Need to find the right function for this. Right now just return the fips"""
    return str(fips)


def _generate_input_data(
    fips, include_testing_correction=True, include_deaths=True, figure_collector=None
):
    """
    Allow the RtInferenceEngine to be agnostic to aggregation level by handling the loading first

    include_testing_correction: bool
        If True, include a correction for testing increases and decreases.
    """
    times, observed_new_cases, observed_new_deaths = load_data.load_new_case_data_by_fips(
        fips, t0=InferRtConstants.REF_DATE, include_testing_correction=include_testing_correction
    )

    date = [InferRtConstants.REF_DATE + timedelta(days=int(t)) for t in times]

    df = filter_and_smooth_input_data(
        df=pd.DataFrame(dict(cases=observed_new_cases, deaths=observed_new_deaths), index=date),
        include_deaths=include_deaths,
        figure_collector=figure_collector,
        display_name=fips,
    )
    return df


def filter_and_smooth_input_data(
    df: [pd.DataFrame], display_name, include_deaths=False, figure_collector=None
) -> pd.DataFrame:
    """Do Everything Strange Here Before it Gets to the Inference Engine"""
    MIN_CUMULATIVE_COUNTS = dict(cases=20, deaths=10)
    MIN_INCIDENT_COUNTS = dict(cases=5, deaths=5)

    dates = df.index
    # Apply Business Logic To Filter Raw Data
    for column in ["cases", "deaths"]:
        requirements = [  # All Must Be True
            df[column].count() > InferRtConstants.MIN_TIMESERIES_LENGTH,
            df[column].sum() > MIN_CUMULATIVE_COUNTS[column],
            df[column].max() > MIN_INCIDENT_COUNTS[column],
        ]
        # Now Apply Input Outlier Detection and Smoothing
        filtered = utils.replace_outliers(df[column], log=rt_log)
        # TODO find way to indicate which points filtered in figure below

        assert len(filtered) == len(df[column])
        smoothed = filtered.rolling(
            InferRtConstants.COUNT_SMOOTHING_WINDOW_SIZE,
            win_type="gaussian",
            min_periods=InferRtConstants.COUNT_SMOOTHING_KERNEL_STD,
            center=True,
        ).mean(std=InferRtConstants.COUNT_SMOOTHING_KERNEL_STD)
        # TODO: Only start once non-zero to maintain backwards compatibility?

        # Check if the Post Smoothed Meets the Requirements
        requirements.append(smoothed.max() > MIN_INCIDENT_COUNTS[column])

        # Check include_deaths Flag
        if column == "deaths" and not include_deaths:
            requirements.append(False)
        else:
            requirements.append(True)

        if all(requirements):
            if column == "cases":
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)  # plt.axes
                ax.set_yscale("log")
                chart_min = max(0.1, smoothed.min())
                ax.set_ylim((chart_min, df[column].max()))
                plt.scatter(
                    dates[-len(df[column]) :],
                    df[column],
                    alpha=0.3,
                    label=f"Smoothing of: {column}",
                )
                plt.plot(dates[-len(df[column]) :], smoothed)
                plt.grid(True, which="both")
                plt.xticks(rotation=30)
                plt.xlim(min(dates[-len(df[column]) :]), max(dates) + timedelta(days=2))

                if not figure_collector:
                    plot_path = get_run_artifact_path(display_name, RunArtifact.RT_SMOOTHING_REPORT)
                    plt.savefig(plot_path, bbox_inches="tight")
                    plt.close(fig)
                else:
                    figure_collector["1_smoothed_cases"] = fig

            df[column] = smoothed
        else:
            df = df.drop(columns=column, inplace=False)
            rt_log.info("Dropping:", columns=column, requirements=requirements)

    return df


class RtInferenceEngine:
    """
    This class extends the analysis of Bettencourt et al to include mortality data in a
    pseudo-non-parametric inference of R_t.

    Parameters
    ----------
    data: DataFrame
        DataFrame with a Date index and at least one "cases" column.
    include_deaths: bool
        If True, include the deaths timeseries in the calculation. XCorrelated and Averaged
    display_name: str
        Needed for Figures. Should just return figures along with dataframe and then deal with title
        and save location somewhere downstream.
    fips: str
        Just used for output paths. Should remove with display_name later.
    """

    def __init__(self, data, display_name, fips, include_deaths=False, figure_collector=None):

        self.dates = data.index
        self.cases = data.cases if "cases" in data else None
        self.deaths = data.deaths if "deaths" in data else None

        self.include_deaths = include_deaths
        self.display_name = display_name
        self.fips = fips
        self.figure_collector = figure_collector

        # Load the InferRtConstants (TODO: turn into class constants)
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
        self.log = structlog.getLogger(Rt_Inference_Target=self.display_name)
        self.log_likelihood = None  # TODO: Add this later. Not in init.
        self.log.info(event="Running:")

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
        timeseries:
            The requested timeseries.
        """
        timeseries_type = TimeseriesType(timeseries_type)

        if timeseries_type is TimeseriesType.NEW_CASES:
            return self.dates, self.cases
        elif timeseries_type is TimeseriesType.NEW_DEATHS:
            return self.dates, self.deaths
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
        # TODO FOR ALEX: Please expand this and describe more clearly the meaning of these variables
        a = self.max_scaling_sigma
        if timeseries_scale == 0:
            b = 1.0
        else:
            b = max(1.0, math.sqrt(self.scale_sigma_from_count / timeseries_scale))

        use_sigma = min(a, b) * self.default_process_sigma

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
        dates, timeseries = self.get_timeseries(timeseries_type=timeseries_type)

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
        loop_idx = 0
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
                current_day=loop_idx,
                current_sigma=current_sigma,
                prev_post_am=posteriors[previous_day].argmax(),
                prior_am=current_prior.argmax(),
                like_am=likelihoods[current_day].argmax(),
                post_am=numerator.argmax(),
            )

            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)
            loop_idx += 1

        self.log_likelihood = log_likelihood

        if plot:
            plotting.plot_posteriors(x=posteriors)  # Returns Figure.
            # The interpreter will handle this as it sees fit. Normal builds never call plot flag.

        start_idx = -len(posteriors.columns)

        return dates[start_idx:], posteriors, start_idx

    def get_available_timeseries(self):
        """
        Determine available timeseries for Rt inference calculation
        with constraints below.


        Returns
        -------
        available_timeseries:
          array of available timeseries saved as TimeseriesType
        """
        available_timeseries = []
        _, cases = self.get_timeseries(TimeseriesType.NEW_CASES.value)
        _, deaths = self.get_timeseries(TimeseriesType.NEW_DEATHS.value)

        if np.sum(cases) > self.min_cases:
            available_timeseries.append(TimeseriesType.NEW_CASES)

        if np.sum(deaths) > self.min_deaths:
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
        available_timeseries = []
        if self.cases is not None:
            available_timeseries.append(TimeseriesType.NEW_CASES)
        if self.deaths is not None:  # We drop deaths in the data loader so don't need to check here
            available_timeseries.append(TimeseriesType.NEW_DEATHS)

        for timeseries_type in available_timeseries:
            # Add Raw Data Output to Output DataFrame
            dates_raw, timeseries_raw = self.get_timeseries(timeseries_type)
            df_raw = pd.DataFrame()
            df_raw["date"] = dates_raw
            df_raw = df_raw.set_index("date")
            df_raw[timeseries_type.value] = timeseries_raw

            df = pd.DataFrame()
            dates, posteriors, start_idx = self.get_posteriors(timeseries_type)
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

                shift_in_days = utils.align_time_series(series_a=series_a, series_b=series_b)

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
                        # Extend death rt signals beyond
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
            if self.figure_collector is None:
                output_path = get_run_artifact_path(self.fips, RunArtifact.RT_INFERENCE_REPORT)
                fig.savefig(output_path, bbox_inches="tight")
            else:
                self.figure_collector["3_Rt_inference"] = fig
        if df_all.empty:
            logging.warning("Inference not possible for fips: %s", self.fips)
        return df_all
