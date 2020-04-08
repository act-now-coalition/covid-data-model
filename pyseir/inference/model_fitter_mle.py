import logging
import iminuit
import numpy as np
import os
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


def get_average_SEIR_parameters(fips):
    """
    Generate the additional fitter candidates from the ensemble generator. This
    has the suppresssion policy and R0 keys removed.

    Returns
    -------
    params: dict
        The average ensemble params.
    """
    SEIR_kwargs = ParameterEnsembleGenerator(fips, N_samples=10000,
                                             t_list=t_list,
                                             suppression_policy=None).get_average_seir_parameters()
    SEIR_kwargs.pop('R0')
    SEIR_kwargs.pop('suppression_policy')
    return SEIR_kwargs


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

    def _fit_seir(R0, t0, eps):
        model = SEIRModel(
            R0=R0,
            suppression_policy=suppression_policies.generate_empirical_distancing_policy(t_list, fips, future_suppression=eps),
            **SEIR_params
        )
        model.run()

        predicted_cases = model.gamma * np.interp(times, t_list + t0, model.results['total_new_infections'])
        predicted_deaths = np.interp(times, t_list + t0, model.results['direct_deaths_per_day'])

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
        if observed_new_deaths.sum() > 5:
            chi2_deaths = np.sum(
                (observed_new_deaths - predicted_deaths) ** 2 / deaths_variance)
        else:
            chi2_deaths = 0
        return chi2_deaths + chi2_cases

    # Note that error def is not right here. We need a realistic error model...
    t0_guess = 50
    m = iminuit.Minuit(_fit_seir, R0=4, t0=t0_guess, eps=.5, error_eps=.2, limit_R0=[1, 8],
                       limit_eps=[0, 2], limit_t0=[-90, 90], error_t0=1, error_R0=1.,
                       errordef=1)
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
    return values


def plot_inferred_result(fit_results):
    """
    Plot the results of an MLE inference
    """
    fips = fit_results['fips']
    county_metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
    times, observed_new_cases, observed_new_deaths = load_data.load_new_case_data_by_fips(fips, t0=ref_date)
    if observed_new_cases.sum() < 5:
        logging.warning(f"{county_metadata['county']} has fewer than 5 cases. Aborting plot.")
        return
    else:
        logging.info(f"Plotting MLE Fits for {county_metadata['county']}")

    R0, t0, eps = fit_results['R0'], fit_results['t0'], fit_results['eps']

    model = SEIRModel(
        R0=R0,
        suppression_policy=suppression_policies.generate_empirical_distancing_policy(t_list, fips, future_suppression=eps),
        **get_average_SEIR_parameters(fit_results['fips'])
    )
    model.run()

    data_dates = [ref_date + timedelta(days=t) for t in times]
    model_dates = [ref_date + timedelta(days=t + fit_results['t0']) for t in t_list]
    plt.figure(figsize=(10, 8))
    plt.errorbar(data_dates, observed_new_cases, marker='o', linestyle='', label='Observed Cases Per Day')
    plt.errorbar(data_dates, observed_new_deaths, yerr=np.sqrt(observed_new_deaths), marker='o', linestyle='', label='Observed Deaths')
    plt.plot(model_dates, model.results['total_new_infections'], label='Estimated Total New Infections Per Day')
    plt.plot(model_dates, model.gamma * model.results['total_new_infections'], label='Symptomatic Model Cases Per Day')
    plt.plot(model_dates, model.results['direct_deaths_per_day'], label='Model Deaths Per Day')
    plt.yscale('log')
    plt.ylim(.9e0)
    plt.xlim(data_dates[0], data_dates[-1] + timedelta(days=90))

    plt.xticks(rotation=30)
    plt.legend(loc=1)
    plt.grid(which='both', alpha=.3)
    plt.title(county_metadata['county'])
    for i, (k, v) in enumerate(fit_results.items()):
        if k not in ('fips', 't0_date', 'county', 'state'):
            plt.text(.025, .97 - 0.04 * i, f'{k}={v:1.3f}',
                     transform=plt.gca().transAxes, fontsize=12)
        else:
            plt.text(.025, .97 - 0.04 * i, f'{k}={v}',
                     transform=plt.gca().transAxes, fontsize=12)

    output_file = os.path.join(
        OUTPUT_DIR, fit_results['state'].title(), 'reports',
        f'{fit_results["state"]}__{fit_results["county"]}__{fit_results["fips"]}__mle_fit_results.pdf')
    plt.savefig(output_file)


def run_state(state):
    """
    Run the fitter for each county in a state.

    Parameters
    ----------
    state: str
        State to run against.
    """
    df = load_data.load_county_metadata()
    all_fips = df[df['state'].str.lower() == state.lower()].fips

    p = Pool()
    fit_results = p.map(fit_county_model, all_fips)

    output_file = os.path.join(OUTPUT_DIR, state.title(), 'data', f'summary_{state}__mle_fit_results.json')
    pd.DataFrame(fit_results).to_json(output_file)

    p.map(plot_inferred_result, fit_results)
    p.close()


if __name__ == '__main__':
    fips = '06075'
    values = fit_county_model(fips)
    plot_inferred_result(values)
