from typing import Optional
from dataclasses import dataclass
from datetime import timedelta
import structlog


import numpy as np
import numba
import math
import pandas as pd
from datapublic.common_fields import CommonFields
from scipy import stats as sps
from matplotlib import pyplot as plt

from libs.datasets import combined_datasets
from libs import pipeline

# `timeseries` is used as a local name in this file, complicating importing it as a module name.
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from pyseir import load_data
from pyseir.utils import RunArtifact
import pyseir.utils
from pyseir.rt.constants import InferRtConstants
from pyseir.rt import plotting, utils

rt_log = structlog.get_logger(__name__)


SQRT2PI = math.sqrt(2.0 * math.pi)


@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)], fastmath=True)
def normal_pdf(x, mean, std_deviation):
    """Probability density function at `x` of a normal distribution.

    Args:
        x: Value
        mean: Mean of distribution
        std_deviation: Standard deviation of distribution.
    """
    u = (x - mean) / std_deviation
    return math.exp(-0.5 * u ** 2) / (SQRT2PI * std_deviation)


@numba.njit(fastmath=True)
def pdf_vector(x, loc, scale):
    """Replacement for scipy pdf function."""
    array = np.empty((x.size, loc.size))
    for i, a in enumerate(x):
        for j, b in enumerate(loc):
            array[i, j] = normal_pdf(a, b, scale)

    return array


@dataclass(frozen=True)
class RegionalInput:
    _combined_data: OneRegionTimeseriesDataset

    @property
    def region(self):
        return self._combined_data.region

    @property
    def display_name(self) -> str:
        return str(self.region)

    @property
    def timeseries(self) -> OneRegionTimeseriesDataset:
        return self._combined_data

    @staticmethod
    def from_regional_data(dataset: OneRegionTimeseriesDataset) -> "RegionalInput":
        return RegionalInput(_combined_data=dataset)

    @staticmethod
    def from_region(region: pipeline.Region, load_demographics: bool = True) -> "RegionalInput":
        return RegionalInput(
            _combined_data=combined_datasets.RegionalData.from_region(
                region, load_demographics=load_demographics
            ).timeseries
        )

    @staticmethod
    def from_fips(fips: str) -> "RegionalInput":
        return RegionalInput.from_region(pipeline.Region.from_fips(fips))


def run_rt(
    regional_input: RegionalInput,
    include_testing_correction: bool = False,
    figure_collector: Optional[list] = None,
) -> pd.DataFrame:
    """Entry Point for Infer Rt

    Returns an empty DataFrame if inference was not possible.
    """

    # Generate the Data Packet to Pass to RtInferenceEngine
    smoothed_cases = _generate_input_data(
        regional_input=regional_input,
        include_testing_correction=include_testing_correction,
        figure_collector=figure_collector,
    )
    if smoothed_cases is None:
        rt_log.warning(
            event="Infer Rt Skipped. No Data Passed Filter Requirements:",
            region=regional_input.display_name,
        )
        return pd.DataFrame()

    # Save a reference to instantiated engine (eventually I want to pull out the figure
    # generation and saving so that I don't have to pass a display_name and fips into the class
    engine = RtInferenceEngine(
        smoothed_cases, display_name=regional_input.display_name, regional_input=regional_input,
    )

    # Generate the output DataFrame (consider renaming the function infer_all to be clearer)
    output_df = engine.infer_all()

    return output_df


def _generate_input_data(
    regional_input: RegionalInput,
    include_testing_correction: bool,
    figure_collector: Optional[list],
) -> Optional[pd.Series]:
    """
    Allow the RtInferenceEngine to be agnostic to aggregation level by handling the loading first

    include_testing_correction: bool
        If True, include a correction for testing increases and decreases.
    """
    # TODO: Outlier Removal Before Test Correction
    try:
        times, observed_new_cases = load_data.calculate_new_case_data_by_region(
            regional_input.timeseries,
            t0=InferRtConstants.REF_DATE,
            include_testing_correction=include_testing_correction,
        )
    except AssertionError as e:
        rt_log.exception(
            event="An AssertionError was raised in the loading of the data for the calculation of "
            "the Infection Rate Metric",
            region=regional_input.display_name,
        )
        return None

    date = [InferRtConstants.REF_DATE + timedelta(days=int(t)) for t in times]

    observed_new_cases = pd.Series(observed_new_cases, index=date)

    observed_new_cases = filter_and_smooth_input_data(
        observed_new_cases,
        date,
        regional_input.region,
        figure_collector,
        rt_log.new(region=regional_input.display_name),
    )
    return observed_new_cases


def filter_and_smooth_input_data(
    cases: pd.Series,
    dates: list,
    region: pipeline.Region,
    figure_collector: Optional[list],
    log: structlog.BoundLoggerBase,
) -> Optional[pd.Series]:
    """Do Filtering Here Before it Gets to the Inference Engine"""
    MIN_CUMULATIVE_CASE_COUNT = 20
    MIN_INCIDENT_CASE_COUNT = 5

    requirements = [  # All Must Be True
        cases.count() > InferRtConstants.MIN_TIMESERIES_LENGTH,
        cases.sum() > MIN_CUMULATIVE_CASE_COUNT,
        cases.max() > MIN_INCIDENT_CASE_COUNT,
    ]
    smoothed = cases.rolling(
        InferRtConstants.COUNT_SMOOTHING_WINDOW_SIZE,
        win_type="gaussian",
        min_periods=InferRtConstants.COUNT_SMOOTHING_KERNEL_STD,
        center=True,
    ).mean(std=InferRtConstants.COUNT_SMOOTHING_KERNEL_STD)
    # TODO: Only start once non-zero to maintain backwards compatibility?

    # Check if the Post Smoothed Meets the Requirements
    requirements.append(smoothed.max() > MIN_INCIDENT_CASE_COUNT)

    if not all(requirements):
        return None

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)  # plt.axes
    ax.set_yscale("log")
    chart_min = max(0.1, smoothed.min())
    ax.set_ylim((chart_min, cases.max()))
    plt.scatter(
        dates[-len(cases) :], cases, alpha=0.3, label=f"Smoothing of: cases",
    )
    plt.plot(dates[-len(cases) :], smoothed)
    plt.grid(True, which="both")
    plt.xticks(rotation=30)
    plt.xlim(min(dates[-len(cases) :]), max(dates) + timedelta(days=2))

    if not figure_collector:
        plot_path = pyseir.utils.get_run_artifact_path(region, RunArtifact.RT_SMOOTHING_REPORT)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
    else:
        figure_collector["1_smoothed_cases"] = fig

    return smoothed


class RtInferenceEngine:
    """
    This class extends the analysis of Bettencourt et al to include mortality data in a
    pseudo-non-parametric inference of R_t.

    Parameters
    ----------
    data: DataFrame
        DataFrame with a Date index and at least one "cases" column.
    display_name: str
        Needed for Figures. Should just return figures along with dataframe and then deal with title
        and save location somewhere downstream.
    regional_input: RegionalInput
        Just used for output paths. Should remove with display_name later.
    """

    def __init__(
        self, cases: pd.Series, display_name, regional_input: RegionalInput, figure_collector=None,
    ):

        self.dates = cases.index
        self.cases = cases

        self.display_name = display_name
        self.regional_input = regional_input
        self.figure_collector = figure_collector

        # Load the InferRtConstants (TODO: turn into class constants)
        self.r_list = InferRtConstants.R_BUCKETS
        self.window_size = InferRtConstants.COUNT_SMOOTHING_WINDOW_SIZE
        self.kernel_std = InferRtConstants.COUNT_SMOOTHING_KERNEL_STD
        self.default_process_sigma = InferRtConstants.DEFAULT_PROCESS_SIGMA
        self.ref_date = InferRtConstants.REF_DATE
        self.confidence_intervals = InferRtConstants.CONFIDENCE_INTERVALS
        self.min_cases = InferRtConstants.MIN_COUNTS_TO_INFER
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

        # Build process matrix using optimized numba pdf function.
        # This function is equivalent to the following call, but runs about 50% faster:
        # process_matrix = sps.norm(loc=self.r_list, scale=use_sigma).pdf(self.r_list[:, None])
        process_matrix = pdf_vector(self.r_list, self.r_list, use_sigma)

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

    def get_posteriors(self, dates, timeseries, plot=False):
        """
        Generate posteriors for R_t.

        Parameters
        ----------
        timeseries: New X per day (cases).
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
        if len(timeseries) == 0:
            self.log.info("empty timeseries, skipping")
            return None, None, None
        else:
            self.log.info("Analyzing posteriors for timeseries")

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
        prior0 = sps.gamma(a=2).pdf(self.r_list)
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
            elif timeseries[current_day] <= 0.4:
                # If the smoothed values for Daily New Cases is less than or equal to 0.4, then reset the prior to the initial
                # In this branch reinit_prior is equal to the initial prior
                # This magic number was decided in consultation between Brett and Chris on 26 May 2021
                # We looked at the impulse response function for the current input smoothing window
                # 6 cases separated by 2 days each has a peak value of 0.41.
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

    def infer_all(self, plot=True) -> pd.DataFrame:
        """
        Infer R_t from all available data sources.

        Parameters
        ----------
        plot: bool
            If True, generate a plot of the inference.

        Returns
        -------
        inference_results: pd.DataFrame
            Columns containing MAP estimates and confidence intervals.
        """
        df_all = None

        df = pd.DataFrame()
        try:
            dates, posteriors, start_idx = self.get_posteriors(self.dates, self.cases)
        except Exception as e:
            rt_log.exception(
                event="Posterior Calculation Error", region=self.regional_input.display_name,
            )
            raise e

        # Note that it is possible for the dates to be missing days
        # This can cause problems when:
        #   1) computing posteriors that assume continuous data (above),
        #   2) when merging data with variable keys
        if posteriors is None:
            return pd.DataFrame()

        df[f"Rt_MAP__new_cases"] = posteriors.idxmax()
        for ci in self.confidence_intervals:
            ci_low, ci_high = self.highest_density_interval(posteriors, ci=ci)

            low_val = 1 - ci
            high_val = ci
            df[f"Rt_ci{int(math.floor(100 * low_val))}__new_cases"] = ci_low
            df[f"Rt_ci{int(math.floor(100 * high_val))}__new_cases"] = ci_high

        df["date"] = dates
        df = df.set_index("date")

        df_all = df

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
            fig = plotting.plot_rt(df=df_all, display_name=self.display_name)
            if self.figure_collector is None:
                output_path = pyseir.utils.get_run_artifact_path(
                    self.regional_input.region, RunArtifact.RT_INFERENCE_REPORT
                )
                fig.savefig(output_path, bbox_inches="tight")
            else:
                self.figure_collector["3_Rt_inference"] = fig
        if df_all.empty:
            self.log.warning("Inference not possible")
        else:
            df_all = df_all.reset_index(drop=False)  # Move date to column from index to column
            df_all[CommonFields.LOCATION_ID] = self.regional_input.region.location_id
        return df_all
