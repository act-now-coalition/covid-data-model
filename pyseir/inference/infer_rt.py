import math
from datetime import datetime, timedelta
import numpy as np
import sentry_sdk
import logging
import pandas as pd
from scipy import stats as sps
from scipy import signal
from matplotlib import pyplot as plt
import us
from pyseir import load_data
from pyseir.utils import AggregationLevel, TimeseriesType
from pyseir.utils import get_run_artifact_path, RunArtifact
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator

log = logging.getLogger(__name__)
NP_SEED = 42


class RtInferenceEngine:
    """
    This class extends the analysis of Kevin Systrom to include mortality
    and hospitalization data in a pseudo-non-parametric inference of R_t.

    Parameters
    ----------
    fips: str
        State or County fips code
    window_size: int
        Size of the sliding Gaussian window to compute. Note that kernel std
        sets the width of the kernel weight.
    kernel_std: int
        Width of the Gaussian kernel.
    r_list: array-like
        Array of R_t to compute posteriors over. Doesn't really need to be
        configured.
    process_sigma: float
        Stdev of the process model. Increasing this allows for larger
        instant deltas in R_t, shrinking it smooths things, but allows for
        less rapid change. Can be interpreted as the std of the allowed
        shift in R_t day-to-day.
    ref_date:
        Reference date to compute from.
    confidence_intervals: list(float)
        Confidence interval to compute. 0.95 would be 90% credible
        intervals from 5% to 95%.
    min_cases: int
        Minimum number of cases required to run case level inference. These are
        very conservaively weak filters, but prevent cases of basically zero
        data from introducing pathological results.
    min_deaths: int
        Minimum number of deaths required to run death level inference.
    include_testing_corrections: bool
        If True, include a correction for testing increases and decreases.
    """

    def __init__(
        self,
        fips,
        window_size=14,
        kernel_std=5,
        r_list=np.linspace(0, 10, 501),
        process_sigma=0.05,
        ref_date=datetime(year=2020, month=1, day=1),
        confidence_intervals=(0.68, 0.95),
        min_cases=5,
        min_deaths=5,
        include_testing_correction=False,
    ):
        np.random.seed(
            NP_SEED
        )  # Xcor, used in align_time_series,  has some stochastic FFT elements.
        self.fips = fips
        self.r_list = r_list
        self.window_size = window_size
        self.kernel_std = kernel_std
        self.process_sigma = process_sigma
        self.ref_date = ref_date
        self.confidence_intervals = confidence_intervals
        self.min_cases = min_cases
        self.min_deaths = min_deaths
        self.include_testing_correction = include_testing_correction

        if len(fips) == 2:  # State FIPS are 2 digits
            self.agg_level = AggregationLevel.STATE
            self.state_obj = us.states.lookup(self.fips)
            self.state = self.state_obj.name

            (
                self.times,
                self.observed_new_cases,
                self.observed_new_deaths,
            ) = load_data.load_new_case_data_by_state(
                self.state,
                self.ref_date,
                include_testing_correction=self.include_testing_correction,
            )

            (
                self.hospital_times,
                self.hospitalizations,
                self.hospitalization_data_type,
            ) = load_data.load_hospitalization_data_by_state(
                state=self.state_obj.abbr, t0=self.ref_date
            )
            self.display_name = self.state
        else:
            self.agg_level = AggregationLevel.COUNTY
            self.geo_metadata = (
                load_data.load_county_metadata().set_index("fips").loc[fips].to_dict()
            )
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
            ) = load_data.load_new_case_data_by_fips(
                self.fips,
                t0=self.ref_date,
                include_testing_correction=self.include_testing_correction,
            )
            (
                self.hospital_times,
                self.hospitalizations,
                self.hospitalization_data_type,
            ) = load_data.load_hospitalization_data(self.fips, t0=self.ref_date)

        logging.info(f"Running Rt Inference for {self.display_name}")

        self.case_dates = [ref_date + timedelta(days=int(t)) for t in self.times]
        if self.hospitalization_data_type:
            self.hospital_dates = [ref_date + timedelta(days=int(t)) for t in self.hospital_times]

        self.default_parameters = ParameterEnsembleGenerator(
            fips=self.fips, N_samples=500, t_list=np.linspace(0, 365, 366)
        ).get_average_seir_parameters()

        # Serial period = Incubation + 0.5 * Infections
        self.serial_period = (
            1 / self.default_parameters["sigma"] + 0.5 * 1 / self.default_parameters["delta"]
        )

        # If we only receive current hospitalizations, we need to account for
        # the outflow to reconstruct new admissions.
        if (
            self.hospitalization_data_type
            is load_data.HospitalizationDataType.CURRENT_HOSPITALIZATIONS
        ):
            los_general = self.default_parameters["hospitalization_length_of_stay_general"]
            los_icu = self.default_parameters["hospitalization_length_of_stay_icu"]
            hosp_rate_general = self.default_parameters["hospitalization_rate_general"]
            hosp_rate_icu = self.default_parameters["hospitalization_rate_icu"]
            icu_rate = hosp_rate_icu / hosp_rate_general
            flow_out_of_hosp = self.hospitalizations[:-1] * (
                (1 - icu_rate) / los_general + icu_rate / los_icu
            )
            # We are attempting to reconstruct the cumulative hospitalizations.
            self.hospitalizations = np.diff(self.hospitalizations) + flow_out_of_hosp
            self.hospital_dates = self.hospital_dates[1:]
            self.hospital_times = self.hospital_times[1:]

        self.log_likelihood = None

    def get_timeseries(self, timeseries_type):
        """
        Given a timeseries type, return the dates, times, and hospitalizations.

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
        elif timeseries_type is TimeseriesType.NEW_DEATHS:
            return self.case_dates, self.times, self.observed_new_deaths
        elif timeseries_type in (
            TimeseriesType.NEW_HOSPITALIZATIONS,
            TimeseriesType.CURRENT_HOSPITALIZATIONS,
        ):
            return self.hospital_dates, self.hospital_times, self.hospitalizations

    def apply_gaussian_smoothing(self, timeseries_type, plot=False, smoothed_max_threshold=5):
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
            constant smoothed values having a dispropotionate effect on
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

        # Hospitalizations have a strange effect in the first few data points across many states. Let's just drop those..
        if timeseries_type in (
            TimeseriesType.CURRENT_HOSPITALIZATIONS,
            TimeseriesType.NEW_HOSPITALIZATIONS,
        ):
            dates, times, timeseries = dates[2:], times[:2], timeseries[2:]

        timeseries = pd.Series(timeseries)
        smoothed = (
            timeseries.rolling(
                self.window_size, win_type="gaussian", min_periods=self.kernel_std, center=True
            )
            .mean(std=self.kernel_std)
            .round()
        )

        nonzeros = [idx for idx, val in enumerate(smoothed) if val != 0]

        if smoothed.empty:
            idx_start = 0
        elif max(smoothed) < smoothed_max_threshold:
            # skip the entire array.
            idx_start = len(smoothed)
        else:
            idx_start = nonzeros[0]

        smoothed = smoothed.iloc[idx_start:]
        original = timeseries.loc[smoothed.index]

        if plot:
            plt.scatter(
                dates[-len(original) :],
                original,
                alpha=0.3,
                label=timeseries_type.value.replace("_", " ").title() + "Shifted",
            )
            plt.plot(dates[-len(original) :], smoothed)
            plt.grid(True, which="both")
            plt.xticks(rotation=30)
            plt.xlim(min(dates[-len(original) :]), max(dates) + timedelta(days=2))
            plt.legend()

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
        """
        dates, times, timeseries = self.apply_gaussian_smoothing(timeseries_type)
        if len(timeseries) == 0:
            return None, None, None

        # (1) Calculate Lambda (the Poisson likelihood given the data) based on
        # the observed increase from t-1 cases to t cases.
        lam = timeseries[:-1].values * np.exp((self.r_list[:, None] - 1) / self.serial_period)

        # (2) Calculate each day's likelihood over R_t
        likelihoods = pd.DataFrame(
            data=sps.poisson.pmf(timeseries[1:].values, lam),
            index=self.r_list,
            columns=timeseries.index[1:],
        )

        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(loc=self.r_list, scale=self.process_sigma).pdf(
            self.r_list[:, None]
        )

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)

        # (4) Calculate the initial prior. Gamma mean of 3 over Rt
        prior0 = sps.gamma(a=2.5).pdf(self.r_list)
        prior0 /= prior0.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=self.r_list, columns=timeseries.index, data={timeseries.index[0]: prior0}
        )

        # We said we'd keep track of the sum of the log of the probability
        # of the data for maximum likelihood calculation.
        log_likelihood = 0.0

        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(timeseries.index[:-1], timeseries.index[1:]):
            # (5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]

            # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior

            # (5c) Calcluate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            if denominator == 0:
                # Restart the baysian learning for the remaining series.
                # This is necessary since otherwise NaN values
                # will be inferred for all future days, after seeing
                # a single (smoothed) zero value.
                #
                # We understand that restarting the posteriors with the
                # initial prior may incur a start-up artifact as the posterior
                # restabilizes, but we believe it's the current best
                # solution for municipalities that have smoothed cases and
                # deaths that dip down to zero, but then start to increase
                # again.

                posteriors[current_day] = prior0
            else:
                posteriors[current_day] = numerator / denominator

            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)

        self.log_likelihood = log_likelihood

        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(posteriors, alpha=0.1, color="k")
            plt.grid(alpha=0.4)
            plt.xlabel("$R_t$", fontsize=16)
            plt.title("Posteriors", fontsize=18)
        start_idx = -len(posteriors.columns)

        return dates[start_idx:], times[start_idx:], posteriors

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
        IDX_OF_COUNTS = 2
        cases = self.get_timeseries(TimeseriesType.NEW_CASES.value)[IDX_OF_COUNTS]
        deaths = self.get_timeseries(TimeseriesType.NEW_DEATHS.value)[IDX_OF_COUNTS]
        if self.hospitalization_data_type:
            hosps = self.get_timeseries(TimeseriesType.NEW_HOSPITALIZATIONS.value)[IDX_OF_COUNTS]

        if np.sum(cases) > self.min_cases:
            available_timeseries.append(TimeseriesType.NEW_CASES)

        if np.sum(deaths) > self.min_deaths:
            available_timeseries.append(TimeseriesType.NEW_DEATHS)

        if (
            self.hospitalization_data_type
            is load_data.HospitalizationDataType.CURRENT_HOSPITALIZATIONS
            and len(hosps > 3)
        ):
            # We have converted this timeseries to new hospitalizations.
            available_timeseries.append(TimeseriesType.NEW_HOSPITALIZATIONS)
        elif (
            self.hospitalization_data_type
            is load_data.HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS
            and len(hosps > 3)
        ):
            available_timeseries.append(TimeseriesType.NEW_HOSPITALIZATIONS)

        for timeseries_type in available_timeseries:

            df = pd.DataFrame()
            dates, times, posteriors = self.get_posteriors(timeseries_type)
            if posteriors is not None:
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
                    df_all = df_all.merge(df, left_index=True, right_index=True, how="outer")

                # ------------------------------------------------
                # Compute the indicator lag using the curvature
                # alignment method.
                # ------------------------------------------------
                if (
                    timeseries_type
                    in (TimeseriesType.NEW_DEATHS, TimeseriesType.NEW_HOSPITALIZATIONS)
                    and f"Rt_MAP__{TimeseriesType.NEW_CASES.value}" in df_all.columns
                ):

                    # Go back upto 30 days or the max time series length we have if shorter.
                    last_idx = max(-21, -len(df))
                    series_a = df_all[f"Rt_MAP__{TimeseriesType.NEW_CASES.value}"].iloc[-last_idx:]
                    series_b = df_all[f"Rt_MAP__{timeseries_type.value}"].iloc[-last_idx:]

                    shift_in_days = self.align_time_series(series_a=series_a, series_b=series_b,)

                    df_all[f"lag_days__{timeseries_type.value}"] = shift_in_days
                    log.debug(
                        "Using timeshift of: %s for timeseries type: %s ",
                        shift_in_days,
                        timeseries_type,
                    )
                    # Shift all the columns.
                    for col in df_all.columns:
                        if timeseries_type.value in col:
                            df_all[col] = df_all[col].shift(shift_in_days)
                            # Extend death and hopitalization rt signals beyond
                            # shift to avoid sudden jumps in composit metric.
                            #
                            # N.B interpolate() behaves differently depending on the location
                            # of the missing values: For any nans appearing in between valid
                            # elements of the series, an interpolated value is filled in.
                            # For values at the end of the series, the last *valid* value is used.
                            log.debug("Filling in %s missing values", shift_in_days)
                            df_all[col] = df_all[col].interpolate(
                                limit_direction="forward", method="linear"
                            )

        if df_all is not None and "Rt_MAP__new_deaths" in df_all and "Rt_MAP__new_cases" in df_all:
            df_all["Rt_MAP_composite"] = np.nanmean(
                df_all[["Rt_MAP__new_cases", "Rt_MAP__new_deaths"]], axis=1
            )
            # Just use the Stdev of cases. A correlated quadrature summed error
            # would be better, but is also more confusing and difficult to fix
            # discontinuities between death and case errors since deaths are
            # only available for a subset. Systematic errors are much larger in
            # any case.
            df_all["Rt_ci95_composite"] = df_all["Rt_ci95__new_cases"]

        elif df_all is not None and "Rt_MAP__new_cases" in df_all:
            df_all["Rt_MAP_composite"] = df_all["Rt_MAP__new_cases"]
            df_all["Rt_ci95_composite"] = df_all["Rt_ci95__new_cases"]

        if plot and df_all is not None:
            plt.figure(figsize=(10, 6))

            if "Rt_MAP_composite" in df_all:
                plt.scatter(
                    df_all.index,
                    df_all["Rt_MAP_composite"],
                    alpha=1,
                    s=25,
                    color="yellow",
                    label="Inferred $R_{t}$ Web",
                    marker="d",
                )
            plt.hlines([1.0], *plt.xlim(), alpha=1, color="g")
            plt.hlines([1.1], *plt.xlim(), alpha=1, color="gold")
            plt.hlines([1.3], *plt.xlim(), alpha=1, color="r")

            if "Rt_ci5__new_deaths" in df_all:
                plt.fill_between(
                    df_all.index,
                    df_all["Rt_ci5__new_deaths"],
                    df_all["Rt_ci95__new_deaths"],
                    alpha=0.2,
                    color="firebrick",
                )
                plt.scatter(
                    df_all.index,
                    df_all["Rt_MAP__new_deaths"].shift(periods=shift_deaths),
                    alpha=1,
                    s=25,
                    color="firebrick",
                    label="New Deaths",
                )

            if "Rt_ci5__new_cases" in df_all:
                plt.fill_between(
                    df_all.index,
                    df_all["Rt_ci5__new_cases"],
                    df_all["Rt_ci95__new_cases"],
                    alpha=0.2,
                    color="steelblue",
                )
                plt.scatter(
                    df_all.index,
                    df_all["Rt_MAP__new_cases"],
                    alpha=1,
                    s=25,
                    color="steelblue",
                    label="New Cases",
                    marker="s",
                )

            if "Rt_ci5__new_hospitalizations" in df_all:
                plt.fill_between(
                    df_all.index,
                    df_all["Rt_ci5__new_hospitalizations"],
                    df_all["Rt_ci95__new_hospitalizations"],
                    alpha=0.4,
                    color="darkseagreen",
                )
                plt.scatter(
                    df_all.index,
                    df_all["Rt_MAP__new_hospitalizations"],
                    alpha=1,
                    s=25,
                    color="darkseagreen",
                    label="New Hospitalizations",
                    marker="d",
                )

            plt.hlines([1.0], *plt.xlim(), alpha=1, color="g")
            plt.hlines([1.1], *plt.xlim(), alpha=1, color="gold")
            plt.hlines([1.3], *plt.xlim(), alpha=1, color="r")

            plt.xticks(rotation=30)
            plt.grid(True)
            plt.xlim(df_all.index.min() - timedelta(days=2), df_all.index.max() + timedelta(days=2))
            plt.ylim(-1, 4)
            plt.ylabel("$R_t$", fontsize=16)
            plt.legend()
            plt.title(self.display_name, fontsize=16)

            output_path = get_run_artifact_path(self.fips, RunArtifact.RT_INFERENCE_REPORT)
            plt.savefig(output_path, bbox_inches="tight")
            # plt.close()
        if df_all is None or df_all.empty:
            log.warning("Inference not possible for fips: %s", self.fips)
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
        np.random.seed(NP_SEED)  # Xcor has some stochastic FFT elements.
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
        except Exception as e:
            sentry_sdk.capture_exception(e)
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
        log.error("Emtpy dataframe encountered! No RtInfernce results available for %s", state)
    else:
        df.to_json(output_path)

    # Run the counties.
    if not states_only:
        all_fips = load_data.get_all_fips_codes_for_a_state(state)

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
    state: str
        State to run against.
    states_only: bool
        If True only run the state level.
    """
    if not fips:
        return None

    df = RtInferenceEngine.run_for_fips(fips)
    county_output_file = get_run_artifact_path(fips, RunArtifact.RT_INFERENCE_RESULT)
    if df is not None and not df.empty:
        df.to_json(county_output_file)
