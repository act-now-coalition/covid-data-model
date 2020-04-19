from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats as sps
from matplotlib import pyplot as plt
import us
from pyseir import load_data
from pyseir.utils import AggregationLevel, TimeseriesType
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator


class RtInferenceEngine:

    def __init__(self,
                 fips,
                 window_size=7,
                 kernel_std=2,
                 r_list=np.linspace(0, 10, 1001),
                 process_sigma=0.15,
                 serial_period=7,
                 ref_date=datetime(year=2020, month=1, day=1),
                 confidence_intervals=[0.68, 0.75, 0.95]):
        """

        Parameters
        ----------
        fips
        window_size
        kernel_std
        r_list
        process_sigma
        serial_period
        ref_date
        output_intervals: float
            Confidence interval to compute. 0.95 would be 95% credible
            intervals.
        """

        self.fips = fips
        self.r_list = r_list
        self.window_size = window_size
        self.kernel_std = kernel_std
        self.process_sigma = process_sigma
        self.serial_period = serial_period
        self.gamma = 1 / self.serial_period
        self.ref_date = ref_date
        self.confidence_intervals = confidence_intervals

        if len(fips) == 2:  # State FIPS are 2 digits
            self.agg_level = AggregationLevel.STATE
            self.state_obj = us.states.lookup(self.fips)
            self.state = self.state_obj.name
            self.geo_metadata = load_data.load_county_metadata_by_state(self.state).loc[self.state].to_dict()

            self.times, self.observed_new_cases, self.observed_new_deaths = \
                load_data.load_new_case_data_by_state(self.state, self.ref_date)

            self.hospital_times, self.hospitalizations, self.hospitalization_data_type = \
                load_data.load_hospitalization_data_by_state(self.state_obj.abbr, t0=self.ref_date)
            self.display_name = self.state
        else:
            self.agg_level = AggregationLevel.COUNTY
            self.geo_metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
            self.state = self.geo_metadata['state']
            self.state_obj = us.states.lookup(self.state)
            self.county = self.geo_metadata['county']
            if self.county:
                self.display_name = self.county + ', ' + self.state
            else:
                self.display_name = self.state
            # TODO Swap for new data source.
            self.times, self.observed_new_cases, self.observed_new_deaths = \
                load_data.load_new_case_data_by_fips(self.fips, t0=self.ref_date)
            self.hospital_times, self.hospitalizations, self.hospitalization_data_type = \
                load_data.load_hospitalization_data(self.fips, t0=self.ref_date)

        self.case_dates = [ref_date + timedelta(days=int(t)) for t in self.times]
        if self.hospitalization_data_type:
            self.hospital_dates = [ref_date + timedelta(days=int(t)) for t in self.hospital_times]

        self.default_parameters = ParameterEnsembleGenerator(
            fips=self.fips,
            N_samples=500,
            t_list=np.linspace(0, 365, 366)
        ).get_average_seir_parameters()

        # If we only receive current hospitalizations, we need to account for
        # the outflow to reconstruct new admissions.
        if self.hospitalization_data_type is load_data.HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
            los_general = self.default_parameters['hospitalization_length_of_stay_general']
            los_icu = self.default_parameters['hospitalization_length_of_stay_general']
            hosp_rate_general = self.default_parameters['hospitalization_rate_general']
            hosp_rate_icu = self.default_parameters['hospitalization_rate_icu']
            icu_rate = hosp_rate_icu / hosp_rate_general
            flow_out_of_hosp = self.hospitalizations[:-1] * ((1 - icu_rate) / los_general + icu_rate / los_icu)
            # We are attempting to reconstruct the cumulative hospitalizations.
            self.hospitalizations = np.diff(self.hospitalizations) + flow_out_of_hosp
            self.hospital_dates = self.hospital_dates[1:]
            self.hospital_times = self.hospital_times[1:]

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
        elif timeseries_type in (TimeseriesType.NEW_HOSPITALIZATIONS, TimeseriesType.CURRENT_HOSPITALIZATOINS):
            return self.hospital_dates, self.hospital_times, self.hospitalizations

    def apply_gaussian_smoothing(self, timeseries_type, plot=False):
        """
        Apply a rolling Gaussian window to smooth the data. This signature and
        returns match get_time_series, but will return a subset of the input
        time-series starting at the first non-zero value.

        Parameters
        ----------
        timeseries_type: TimeseriesType
            Which type of time-series to use.

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
        timeseries = pd.Series(timeseries)
        smoothed = timeseries.rolling(self.window_size, win_type='gaussian',
                                     min_periods=self.kernel_std, center=True)\
                             .mean(std=self.kernel_std)\
                             .round()

        zeros = smoothed.index[smoothed.eq(0)]
        if len(zeros) == 0:
            idx_start = 0
        else:
            last_zero = zeros.max()
            idx_start = smoothed.index.get_loc(last_zero) + 1
        smoothed = smoothed.iloc[idx_start:]
        original = timeseries.loc[smoothed.index]

        if plot:
            plt.scatter(dates[-len(original):], original, alpha=0.3, label=timeseries_type.value.replace('_', ' ').title() + 'Shifted')
            plt.plot(dates[-len(original):], smoothed)
            plt.grid(True, which='both')
            plt.xticks(rotation=30)
            plt.xlim(min(dates[-len(original):]), max(dates) + timedelta(days=2))
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
        posterior_pdfs = posteriors.values.cumsum(axis=0)
        low_idx_list = np.argmin(np.abs(posterior_pdfs - (1 - ci)), axis=0)
        high_idx_list = np.argmin(np.abs(posterior_pdfs - ci), axis=0)
        ci_low = self.r_list[low_idx_list]
        ci_high = self.r_list[high_idx_list]
        return ci_low, ci_high

    def get_posteriors(self, timeseries_type, plot=False):
        """
        Generate posteriors for R_t.

        Parameters
        ----------
        timeseries_type: TimeseriesType
            New X per day (cases, deaths etc).

        Returns
        -------
        dates: array-like
            Input data over a subset of indices available after windowing.
        times: array-like
            Output integers since the reference date.
        posteriors: pd.DataFrame
            Posterior estimiates for each timestamp with non-zero data.
        """
        timeseries_type = TimeseriesType(timeseries_type)

        dates, times, timeseries = self.apply_gaussian_smoothing(timeseries_type)

        # (1) Calculate Lambda
        lam = timeseries[:-1].values * np.exp(self.gamma * (self.r_list[:, None] - 1))

        # (2) Calculate each day's likelihood
        likelihoods = pd.DataFrame(
            data=sps.poisson.pmf(timeseries[1:].values, lam),
            index=self.r_list,
            columns=timeseries.index[1:])

        # (3) Create the Gaussian Matrix
        process_matrix = sps.norm(loc=self.r_list, scale=self.process_sigma).pdf(self.r_list[:, None])

        # (3a) Normalize all rows to sum to 1
        process_matrix /= process_matrix.sum(axis=0)

        # (4) Calculate the initial prior
        prior0 = sps.gamma(a=4).pdf(self.r_list)
        prior0 /= prior0.sum()

        # Create a DataFrame that will hold our posteriors for each day
        # Insert our prior as the first posterior.
        posteriors = pd.DataFrame(
            index=self.r_list,
            columns=timeseries.index,
            data={timeseries.index[0]: prior0}
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
            posteriors[current_day] = numerator / denominator

            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)

        self.log_likelihood = log_likelihood

        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(posteriors, alpha=0.1, color='k')
            plt.grid(alpha=0.4)
            plt.xlabel('$R_t$', fontsize=16)
            plt.title('Posteriors', fontsize=18)
        start_idx = -len(posteriors.columns)
        return dates[start_idx:], times[start_idx:], posteriors

    def infer_all(self, plot=False, shift_deaths=0):
        """
        Infer R_t from all available data sources.

        Returns
        -------
        inference_results: pd.DataFrame
        """
        df_all = None
        available_timeseries = [TimeseriesType.NEW_CASES, TimeseriesType.NEW_DEATHS]
        if self.hospitalization_data_type is load_data.HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
            available_timeseries.append(TimeseriesType.CURRENT_HOSPITALIZATIONS)
        elif self.hospitalization_data_type is load_data.HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
            available_timeseries.append(TimeseriesType.NEW_HOSPITALIZATIONS)

        for timeseries_type in available_timeseries:

            df = pd.DataFrame()
            dates, times, posteriors = self.get_posteriors(timeseries_type)
            df[f'Rt_MAP__{timeseries_type.value}'] = posteriors.idxmax()
            for ci in self.confidence_intervals:
                ci_low, ci_high = self.highest_density_interval(posteriors, ci=ci)
                df[f'Rt_ci{int(100 * (1 - ci))}__{timeseries_type.value}'] = ci_low
                df[f'Rt_ci{int(100 * ci)}__{timeseries_type.value}'] = ci_high

            df['date'] = dates
            df = df.set_index('date')

            if df_all is None:
                df_all = df
            else:
                df_all = df_all.merge(df, left_index=True, right_index=True, how='outer')

        if plot:
            plt.figure(figsize=(10, 6))

            plt.fill_between(df_all.index,  df_all['Rt_ci5__new_deaths'],  df_all['Rt_ci95__new_deaths'],
                             alpha=.2, color='firebrick')
            plt.scatter(df_all.index, df_all['Rt_MAP__new_deaths'].shift(periods=shift_deaths),
                        alpha=1, s=25, color='firebrick', label='New Deaths')

            plt.fill_between(df_all.index, df_all['Rt_ci5__new_cases'], df_all['Rt_ci95__new_cases'],
                             alpha=.2, color='steelblue')
            plt.scatter(df_all.index, df_all['Rt_MAP__new_cases'],
                        alpha=1, s=25, color='steelblue', label='New Cases', marker='s')

            if self.hospitalization_data_type:
                plt.fill_between(df_all.index, df_all['Rt_ci5__new_hospitalizations'], df_all['Rt_ci95__new_hospitalizations'],
                                 alpha=.4, color='darkseagreen')
                plt.scatter(df_all.index, df_all['Rt_MAP__new_hospitalizations'],
                            alpha=1, s=25, color='darkseagreen', label='New Hospitalizations', marker='d')

            plt.hlines([1.0], *plt.xlim(), alpha=1, color='g')
            plt.hlines([1.1], *plt.xlim(), alpha=1, color='gold')
            plt.hlines([1.3], *plt.xlim(), alpha=1, color='r')

            plt.xticks(rotation=30)
            plt.grid(True)
            plt.xlim(df_all.index.min() - timedelta(days=2), df_all.index.max() + timedelta(days=2))
            plt.ylim(0, 5)
            plt.ylabel('$R_t$', fontsize=16)
            plt.legend()
            plt.title(self.display_name, fontsize=16)

        return df_all
