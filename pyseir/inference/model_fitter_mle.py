import logging
import iminuit
import numpy as np
import os
import us
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from multiprocessing import Pool
from pyseir.models import suppression_policies
from pyseir import load_data, OUTPUT_DIR
from pyseir.models.seir_model import SEIRModel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator

t_list = np.linspace(0, 1000, 1001)
ref_date = datetime(year=2020, month=1, day=1)
frac_counties_with_seed_infection = 0.5
min_deaths = 5   # minimum number of deaths for chi2 calculation and plot.

def get_average_SEIR_parameters(fips):
    """
    Generate the additional fitter candidates from the ensemble generator. This
    has the suppresssion policy and R0 keys removed.
    Returns
    -------
    params: dict
        The average ensemble params.
    """
    SEIR_kwargs = ParameterEnsembleGenerator(fips, N_samples=100,
                                             t_list=t_list,
                                             suppression_policy=None).get_average_seir_parameters()
    SEIR_kwargs.pop('R0')
    SEIR_kwargs.pop('suppression_policy')
    return SEIR_kwargs

def fit_seir(R0, t0, eps, times,
             by='fips', fips=None, state=None,
             observed_new_cases=None, observed_new_deaths=None,
             SEIR_params=None,
             suppression_policy_params=None):
    """
    Fit SEIR model by MLE.
    Parameters
    ----------
    R0: float
        Basic reproduction number
    t0: float
        Epidemic starting time.
    eps:
        Fraction of reduction in contact rates as result of  to suppression
        policy projected into future.
    times: np.array
        Time since t0 of observed new cases or new deaths.
    by: str
        Level of district to fit the seir model for, should be either 'fips'
        or 'state'.
    fips: str
        County fips code.
    state: str
        Full state name.
    observed_new_cases: np.array
        Observed new cases.
    observed_new_deaths: np.array
        Observed new deaths.
    SEIR_params: dict
        Parameters to pass to SEIR model.
    suppression_policy_params: dict
        Parameters to pass to suppression policy model.

    Returns
    -------
      : float
        Chi square of fitting model to observed cases and deaths.
    """

    if by == 'fips':
        suppression_policy = \
        suppression_policies.generate_empirical_distancing_policy(
            fips=fips, future_suppression=eps, **suppression_policy_params)
    elif by == 'state':
        suppression_policy = \
        suppression_policies.generate_empirical_distancing_policy_by_state(
            state=state, future_suppression=eps, **suppression_policy_params)

    model = SEIRModel(
        R0=R0, suppression_policy=suppression_policy,
        **SEIR_params)
    model.run()

    predicted_cases = model.gamma * np.interp(times, t_list + t0, model.results[
        'total_new_infections'])
    predicted_deaths = np.interp(times, t_list + t0, model.results[
        'total_deaths_per_day'])

    # Assume the error on the case count could be off by a massive factor 50.
    # Basically we don't want to use it if there appreciable mortality data available.
    # Longer term there is a better procedure.
    cases_variance = 1e10 * observed_new_cases.copy() ** 2  # Make the stdev N times larger x the number of cases
    deaths_variance = observed_new_deaths.copy()  # Poisson dist error

    # Zero inflated poisson Avoid floating point errors..
    cases_variance[cases_variance == 0] = 1e10
    deaths_variance[deaths_variance == 0] = 1e10

    # Compute Chi2
    chi2_cases = np.sum((observed_new_cases - predicted_cases) ** 2 / cases_variance)
    if observed_new_deaths.sum() > min_deaths:
        chi2_deaths = np.sum(
            (observed_new_deaths - predicted_deaths) ** 2 / deaths_variance)
    else:
        chi2_deaths = 0
    return chi2_deaths + chi2_cases


def fit_county_model(fips):
    """
    Fit the county's current trajectory, using the existing measures. We fit
    only to mortality data if available, else revert to case data.
    We assume a poisson process generates mortalities at a rate defined by the
    underlying dynamical model.
    TODO @ EC: Add hospitalization data when available.
    Parameters
    ----------
    fips: str
        County fips.
    Returns
    -------
    fit_values: dict
        Optimal values from the fitter.
    """
    county_metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
    times, observed_new_cases, observed_new_deaths = load_data.load_new_case_data_by_fips(fips, t0=ref_date)

    logging.info(f'Fitting MLE model to {county_metadata["county"]}, {county_metadata["state"]}')
    SEIR_params = get_average_SEIR_parameters(fips)
    suppression_policy_params = dict(t_list=t_list,
                                     reference_start_date=ref_date)

    # Note that error def is not right here. We need a realistic error model...
    t0_guess = 50
    _fit_seir = lambda R0, t0, eps: fit_seir(R0=R0, t0=t0, eps=eps, times=times,
                                             by='fips', fips=fips,
                                             observed_new_cases=observed_new_cases,
                                             observed_new_deaths=observed_new_deaths,
                                             suppression_policy_params=suppression_policy_params,
                                             SEIR_params=SEIR_params
                                              )

    m = iminuit.Minuit(_fit_seir,
                       R0=4, t0=t0_guess, eps=.5, error_eps=.2,
                       limit_R0=[1, 8],
                       limit_eps=[0, 2], limit_t0=[-90, 90], error_t0=1, error_R0=1.,
                       errordef=1)

    # run MIGRAD algorithm for optimization.
    # for details refer: https://root.cern/root/html528/TMinuit.html
    # TODO @ Xinyu: add lines to check if minuit optimization result is valid.
    m.migrad()

    values = dict(fips=fips, **dict(m.values))
    if np.isnan(values['t0']):
        logging.error(f'Could not compute MLE values for county {county_metadata["county"]}, {county_metadata["state"]}')
        values['t0_date'] = ref_date + timedelta(days=t0_guess)
    else:
        values['t0_date'] = ref_date + timedelta(days=values['t0'])
    values['Reff_current'] = values['R0'] * values['eps']
    values['observed_total_deaths'] = np.sum(observed_new_deaths)
    values['county'] = county_metadata['county']
    values['state'] = county_metadata['state']
    values['total_population'] = county_metadata['total_population']
    values['population_density'] = county_metadata['population_density']
    values['I_initial'] = SEIR_params['I_initial']
    values['A_initial'] = SEIR_params['A_initial']

    # # TODO @ Xinyu: test this after the slow inference is resolved
    # plot_optimization_results(m, by='fips', fit_results=values)

    return values


def fit_state_model(state):
    """
    Fit the state's current trajectory, using the existing measures. We fit
    only to mortality data if available, else revert to case data.
    We assume a poisson process generates mortalities at a rate defined by the
    underlying dynamical model.
    TODO @ EC: Add hospitalization data when available.
    Parameters
    ----------
    state: str
        State full name.
    Returns
    -------
    fit_values: dict
        Optimal values from the fitter.
    """

    state_metadata = load_data.load_county_metadata_by_state(state) \
                              .loc[state] \
                              .to_dict()

    times, observed_new_cases, observed_new_deaths = \
        load_data.load_new_case_data_by_state(state, ref_date)

    logging.info(f'Fitting MLE model to {state}')
    SEIR_params = get_average_SEIR_parameters(us.states.lookup(state).fips)
    SEIR_params['I_initial'] = \
        SEIR_params['I_initial'] * len(state_metadata['fips']) * frac_counties_with_seed_infection
    SEIR_params['A_initial'] = \
        SEIR_params['A_initial'] * len(state_metadata['fips']) * frac_counties_with_seed_infection
    suppression_policy_params = dict(t_list=t_list,
                                     reference_start_date=ref_date)

    # Note that error def is not right here. We need a realistic error model...
    t0_guess = 50
    _fit_seir = lambda R0, t0, eps: fit_seir(R0=R0, t0=t0, eps=eps, times=times,
                                             by='state', state=state,
                                             observed_new_cases=observed_new_cases,
                                             observed_new_deaths=observed_new_deaths,
                                             suppression_policy_params=suppression_policy_params,
                                             SEIR_params=SEIR_params
                                             )
    m = iminuit.Minuit(_fit_seir,
                       R0=4, t0=t0_guess, eps=0.5,
                       error_eps=.2, error_t0=1, error_R0=1., errordef=1,
                       limit_R0=[1, 8], limit_eps=[0, 2], limit_t0=[-90, 90])

    # run MIGRAD algorithm for optimization.
    # for details refer: https://root.cern/root/html528/TMinuit.html
    # TODO @ Xinyu: add lines to check if minuit optimization result is valid.
    m.migrad()

    values = dict(**dict(m.values))
    if np.isnan(values['t0']):
        logging.error(f'Could not compute MLE values for state {state_metadata["county"]}')
        values['t0_date'] = ref_date + timedelta(days=t0_guess)
    else:
        values['t0_date'] = ref_date + timedelta(days=values['t0'])
    values['Reff_current'] = values['R0'] * values['eps']
    values['observed_total_deaths'] = np.sum(observed_new_deaths)
    values['state'] = state
    values['total_population'] = state_metadata['total_population']
    values['population_density'] = state_metadata['population_density']
    values['I_initial'] = SEIR_params['I_initial']
    values['A_initial'] = SEIR_params['A_initial']

    # TODO @ Xinyu: test this after the slow inference is resolved
    # plot_optimization_results(m, by='state', fit_results=values)

    return values

def plot_optimization_results(m, by, fit_results):
    """
    Plot parameter profile likelihood and contours.
    """

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    m.draw_profile("R0")
    plt.subplot(2, 2, 2)
    m.draw_profile("t0")
    plt.subplot(2, 2, 3)
    m.draw_profile("eps")
    plt.subplot(2, 2, 4)
    m.draw_mncontour('R0', 't0', nsigma=4)

    if by == 'fips':
        output_file = os.path.join(
            OUTPUT_DIR, fit_results['state'].title(), 'reports',
            f'{fit_results["state"]}__{fit_results["county"]}__{fit_results["fips"]}__mle_fit_contours.pdf')
    elif by == 'state':
        output_file = os.path.join(
            OUTPUT_DIR, fit_results['state'].title(), 'reports', f'{fit_results["state"]}__mle_fit_contours.pdf')

    plt.savefig(output_file)

def plot_fitting_results(by,
                         metadata,
                         times,
                         fit_results,
                         observed_new_cases,
                         observed_new_deaths,
                         model):
    """
    Plotting model fitting results.
    """

    data_dates = [ref_date + timedelta(days=t) for t in times]
    model_dates = [ref_date + timedelta(days=t + fit_results['t0']) for t in
                   t_list]
    plt.figure(figsize=(10, 8))
    plt.errorbar(data_dates, observed_new_cases, marker='o', linestyle='',
                 label='Observed Cases Per Day')
    plt.errorbar(data_dates, observed_new_deaths,
                 yerr=np.sqrt(observed_new_deaths), marker='o', linestyle='',
                 label='Observed Deaths')
    plt.plot(model_dates, model.results['total_new_infections'],
             label='Estimated Total New Infections Per Day')
    if model.gamma < 1:
        plt.plot(model_dates, model.gamma * model.results['total_new_infections'],
                 label='Symptomatic Model Cases Per Day')
    plt.plot(model_dates, model.results['total_deaths_per_day'],
             label='Model Deaths Per Day')
    plt.yscale('log')
    plt.ylim(.9e0)
    plt.xlim(data_dates[0], data_dates[-1] + timedelta(days=90))

    plt.xticks(rotation=30)
    plt.legend(loc=1)
    plt.grid(which='both', alpha=.3)

    if by == 'fips':
        plt.title(metadata['county'])
    elif by == 'state':
        plt.title(fit_results['state'])

    for i, (k, v) in enumerate(fit_results.items()):
        if k not in ('fips', 't0_date', 'county', 'state'):
            plt.text(.025, .97 - 0.04 * i, f'{k}={v:1.3f}',
                     transform=plt.gca().transAxes, fontsize=12)
        else:
            plt.text(.025, .97 - 0.04 * i, f'{k}={v}',
                     transform=plt.gca().transAxes, fontsize=12)

    if by == 'fips':
        output_file = os.path.join(
            OUTPUT_DIR, fit_results['state'].title(), 'reports',
            f'{fit_results["state"]}__{fit_results["county"]}__{fit_results["fips"]}__mle_fit_results.pdf')
    elif by == 'state':
        output_file = os.path.join(
            OUTPUT_DIR, fit_results['state'].title(), 'reports',
            f'{fit_results["state"]}__mle_fit_results.pdf')
    plt.savefig(output_file)


def plot_inferred_result_county(fit_results):
    """
    Plot the results of an MLE inference at county level.
    """
    fips = fit_results['fips']
    metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
    times, observed_new_cases, observed_new_deaths = \
            load_data.load_new_case_data_by_fips(fips, t0=ref_date)

    if observed_new_cases.sum() < min_deaths:
        logging.warning(f"{metadata['county']} has fewer than {min_deaths} cases. "
                        f"Aborting plot.")
    else:
        logging.info(f"Plotting MLE Fits for {metadata['county']}")

    R0, t0, eps, optimizer = \
        fit_results['R0'], fit_results['t0'], fit_results['eps'], fit_results['optimizer']

    model = SEIRModel(
        R0=R0,
        suppression_policy=suppression_policies.generate_empirical_distancing_policy(
            t_list, fips, future_suppression=eps),
        **get_average_SEIR_parameters(fit_results['fips'])
    )
    model.run()

    plot_fitting_results(by='fips',
                         metadata=metadata,
                         times=times,
                         fit_results=fit_results,
                         observed_new_cases=observed_new_cases,
                         observed_new_deaths=observed_new_deaths,
                         model=model)


def plot_inferred_result_state(fit_results):
    """
    Plot the results of an MLE inference at state level.
    """
    state = fit_results['state']
    metadata = load_data.load_county_metadata_by_state(state) \
                        .loc[state] \
                        .to_dict()
    times, observed_new_cases, observed_new_deaths = \
            load_data.load_new_case_data_by_state(state, ref_date)

    if observed_new_cases.sum() < min_deaths:
        logging.warning(f"{state} has fewer than {min_deaths} cases. Aborting plot.")
        return
    else:
        logging.info(f"Plotting MLE Fits for {state}")

    R0, t0, eps, I_initial, A_initial, optimizer = \
        fit_results['R0'], fit_results['t0'], fit_results['eps'], \
        fit_results['I_initial'], fit_results['A_initial'], fit_results['optimizer']

    SEIR_params = get_average_SEIR_parameters(us.states.lookup(state).fips)
    SEIR_params['I_initial'] = I_initial
    SEIR_params['A_initial'] = A_initial
    model = SEIRModel(
        R0=R0,
        suppression_policy=suppression_policies.generate_empirical_distancing_policy_by_state(
            t_list, state, future_suppression=eps),
            **SEIR_params
    )
    model.run()

    plot_fitting_results(by='state',
                         metadata=metadata,
                         times=times,
                         fit_results=fit_results,
                         observed_new_cases=observed_new_cases,
                         observed_new_deaths=observed_new_deaths,
                         model=model)


def run_state(state, states_only=False):
    """
    Run the fitter for each county in a state.
    Parameters
    ----------
    state: str
        State to run against.
    states_only: bool
        If True only run the state level.
    """

    if not states_only:
        df = load_data.load_county_metadata()
        all_fips = df[df['state'].str.lower() == state.lower()].fips

        p = Pool()
        fit_results = p.map(fit_county_model, all_fips)

        output_file = os.path.join(OUTPUT_DIR, state.title(), 'data', f'summary_{state}__mle_fit_results.json')
        pd.DataFrame(fit_results).to_json(output_file)

        p.map(plot_inferred_result_county, fit_results)
        p.close()

    else:
        fit_results = fit_state_model(state)
        output_file = os.path.join(OUTPUT_DIR, 'pyseir', state.title(), 'data',
                                   f'summary_{state}_state_only__mle_fit_results.json')
        pd.DataFrame(fit_results, index=[state]).to_json(output_file)

        plot_inferred_result_state(fit_results)

if __name__ == '__main__':
    fips = '06075'
    values = fit_county_model(fips)
    plot_inferred_result_county(values)
