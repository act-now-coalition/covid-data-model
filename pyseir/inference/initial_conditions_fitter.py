from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import pandas as pd
import iminuit
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from multiprocessing import Pool
from functools import partial
import seaborn as sns
from pyseir import load_data
from pyseir import OUTPUT_DIR


class InitialConditionsFitter:
    """
    Fit an exponential model to observations assuming a binomial error on
    observations. Identify t0 at the threshold specified.

    This is left in for initial exponential fitting, but generally one should
    use MLE or other fitters which (i) incorporate the full SEIR model to infer
    parameters, (ii) fit to new case data rather than cumulative, and (iii) add
    mortality.

    Parameters
    ----------
    fips: str
        County fips code
    t0_case_count: int
        Case count to infer the start date of.
    start_days_before_t0: int
        After we find the time of t0_case_count, filter observations
        occurring more than this many days before.
    start_days_after_t0: int
        After we find the time of t0_case_count, filter observations
        occurring more than this many days after.
    """

    def __init__(self, fips, t0_case_count=1, start_days_before_t0=2,
                 start_days_after_t0=1000, min_days_required=5):

        self.t0_case_count = t0_case_count
        self.start_days_before_t0 = start_days_before_t0
        self.start_days_after_t0 = start_days_after_t0
        self.min_days_required = min_days_required

        # Load case data
        case_data = load_data.load_county_case_data()
        self.cases = case_data[case_data['fips'] == fips]
        n_days = len(self.cases)
        if n_days < min_days_required:
            raise ValueError(f'Only {n_days} observations for county. Cannot fit.')
        self.fips = fips

        self.county = self.cases.county.values[0]
        self.state = self.cases.state.values[0]

        self.t = (self.cases.date - self.cases.date.min()).dt.days.values
        self.data_start_date = self.cases.date.min()
        self.y = self.cases.cases.values

        self.fit_predictions = None
        self.t0 = None
        self.t0_date = None
        self.reduced_chi2 = None
        self.model_params = None

    @staticmethod
    def exponential_model(norm, t0, scale, t):
        """
        Simple exponential model.
        """
        return norm * np.exp((t - t0) / scale)

    @staticmethod
    def _reduced_chi2(y_pred, y):
        """
        Calculate reduced chi^2.

        Parameters
        ----------
        y_pred: array-like
            Predictions from the model to fit.
        y: array-like
            Observations to fit against.

        Returns
        -------
        reduced_chi2: float
            chi^2 / d.o.f.
        """
        chi2 = (y_pred[y > 0] - y[y > 0]) ** 2 / y[y > 0]
        return np.sum(chi2) / (len(chi2) - 1)

    def exponential_loss(self, norm, t0, scale):
        """Return the reduced chi2 for an exponential fit to teh data"""
        y_pred = self.exponential_model(norm, t0, scale, self.t)
        return self._reduced_chi2(y_pred, self.y)

    def fit_county_initial_conditions(self):
        """
        Determine the initial conditions by fitting an exponential to a set of
        observations.

        Returns
        -------
        : dict
            Fit parameters norm, t0, scale.
        """
        x0 = dict(norm=1, t0=5, scale=20, error_norm=.01, error_t0=.1, error_scale=.01)
        m = iminuit.Minuit(self.exponential_loss, **x0, errordef=0.5)
        fit = m.migrad()
        return {val['name']: val['value'] for val in fit.params}

    def fit(self):
        """
        Use the migrad algorithm to minimize chi2 over the dataset.
        """
        logging.info(f'Fitting {self.county}, {self.state} Initial Conditions')
        self.model_params = self.fit_county_initial_conditions()
        self.fit_predictions = self.exponential_model(**self.model_params, t=self.t)
        self.reduced_chi2 = self.exponential_loss(**self.model_params)

        self.t0_idx = np.argmin(np.abs(self.fit_predictions - self.t0_case_count))
        self.t0 = self.t[self.t0_idx]
        self.t0_date = self.data_start_date + timedelta(days=int(self.t0))

        self.fit_summary = dict(
            model_params=self.model_params,
            t0_date=self.t0_date,
            reduced_chi2=self.reduced_chi2,
        )

    def plot_fit(self):
        """
        Plot the exponential fit.
        """
        plt.figure(figsize=(10, 7))

        plt.errorbar(self.t - self.t0, self.cases.cases, yerr=np.sqrt(self.cases.cases), marker='o', label='Cases')
        plt.plot(self.t - self.t0, self.fit_predictions, label='Best Fit with Filters')
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.ylabel('Count')
        plt.xlabel(f'Time Since {self.t0_case_count} cases predicted')
        plt.text(.1, .8, '$C(t) = %1.3f\ \exp^{(t - %1.3f) / %1.3f}$' % (self.model_params["norm"], self.model_params["t0"], self.model_params["scale"]),
                 transform=plt.gca().transAxes, fontsize=14)

        plt.text(.1, .7, f'{self.t0_case_count} Cases on {self.t0_date.date()}',
                     transform=plt.gca().transAxes, fontsize=14)
        plt.text(.1, .6, f'$\chi^2 /d.o.f.=%1.3f$' % self.reduced_chi2,
                 transform=plt.gca().transAxes, fontsize=14)

        plt.title(f'{self.county} County, {self.state}  FIPS: {self.fips}')
        plt.legend()


def _fit_fips(fips, generate_report=False):
    """
    Fit an exponential growth to county fips.

    Parameters
    ----------
    fips: str
        County fips code.
    generate_report: bool

    Returns
    -------
    fits_values: dict
        Ex. {'model_params': None, 't0_date': None, 'reduced_chi2': None}
    """
    try:
        fitter = InitialConditionsFitter(
            fips=fips,
            t0_case_count=1,
            start_days_before_t0=0,
            start_days_after_t0=1000)

        fitter.fit()
        if generate_report:
            fitter.plot_fit()
            state_dir = os.path.join(OUTPUT_DIR, fitter.state)
            plt.savefig(os.path.join(state_dir, 'reports', f'{fitter.state}__{fitter.county}__{fitter.fips}__t0_fit.pdf'),
                        bbox_inches='tight')
            plt.close()
        return fitter.fit_summary

    except ValueError as e:
        logging.warning(str(e))
        return {'model_params': None, 't0_date': None, 'reduced_chi2': None}


def generate_start_times_for_state(state, generate_report=False):
    """
    Generate imputed start dates for each county.

    Parameters
    ----------
    state: str
        State to model counties of.
    generate_report: bool
        If True, generate summary plots.
    """
    metadata = load_data.load_county_metadata()
    state_dir = os.path.join(OUTPUT_DIR, state)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(os.path.join(state_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(state_dir, 'data'), exist_ok=True)

    logging.info(f'Imputing start times for {state.capitalize()}')
    counties = metadata[metadata['state'].str.lower() == state.lower()].fips
    if len(counties) == 0:
        raise ValueError(f'No entries for state {state}.')

    # Fit exponential model to extract T0.
    f = partial(_fit_fips, generate_report=generate_report)
    p = Pool()
    fips_to_fit_map = {fips: val for fips, val in zip(counties.values, p.map(f, counties.values))}
    p.close()

    # --------------------------------
    # ML to Impute start time for counties with no data based on pop. density
    # -------------------------------
    # Merge in county level metadata.
    county_fits = pd.DataFrame.from_dict(fips_to_fit_map, orient='index').reset_index().rename({'index': 'fips'}, axis=1)
    merged = county_fits.merge(metadata, on='fips')
    merged['days_from_2020_01_01'] = (merged.t0_date - datetime.fromisoformat('2020-01-01')).dt.days

    samples_with_data = merged['days_from_2020_01_01'].notnull()
    samples_with_no_data = merged['days_from_2020_01_01'].isnull()

    X = np.log(merged[['population_density', 'housing_density', 'total_population']][samples_with_data])
    X_predict = np.log(merged[['population_density', 'housing_density', 'total_population']][samples_with_no_data])

    # Test a few regressions
    for estimator in [LinearRegression(), RandomForestRegressor(), BayesianRidge()]:
        cv_result = cross_validate(estimator, X=X, y=merged['days_from_2020_01_01'][samples_with_data], scoring='r2', cv=4)
        logging.info(f'{estimator.__class__.__name__} CV r2: {cv_result["test_score"].mean()}')

    # Train best model and impute the missing times.
    best_model = BayesianRidge()
    best_model.fit(X=X, y=merged['days_from_2020_01_01'][samples_with_data])

    if samples_with_no_data.values.any():
        merged.loc[samples_with_no_data, 'days_from_2020_01_01'] = best_model.predict(X_predict)
        merged.loc[samples_with_no_data, 't0_date'] = datetime.fromisoformat('2020-01-01') \
                                                      + np.array([timedelta(days=t) for t in best_model.predict(X_predict)])

    # Plot doubling time by population density
    merged.loc[samples_with_no_data, 'imputed_start_time'] = True
    merged.loc[samples_with_data, 'imputed_start_time'] = False
    merged.loc[samples_with_data, 'doubling_rate_days'] = np.log(2) * merged['model_params'][samples_with_data].apply(lambda x: x['scale'])
    merged.to_pickle(os.path.join(state_dir, 'data', f'summary__{state}_imputed_start_times.pkl'))

    # Plot population density
    plt.figure(figsize=(14, 4))
    for i, x in enumerate(('population_density', 'housing_density', 'total_population')):
        plt.subplot(1, 3, i + 1)
        plt.title(state)
        sns.jointplot(x=np.log10(merged[x]), y='days_from_2020_01_01', data=merged, kind='reg', height=5)
        plt.xlabel('log10 Population Density')
    plt.savefig(os.path.join(state_dir, 'reports', f'summary__{state}__population_density.pdf'), bbox_inches='tight')
    plt.close()

    # Plot Doubling Rates by distance
    # TODO: Impute doubling time.
    sns.jointplot(np.log10(merged.population_density), merged.doubling_rate_days, kind='reg', height=10)
    plt.xlabel('Log10 Population Density', fontsize=16)
    plt.ylabel('Doubling Time [Days]', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(state_dir, 'reports', f'summary__{state}__doubling_time.pdf'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    generate_start_times_for_state('California')
