import datetime

import numpy as np
import pandas as pd

# odeint might work, moving it out didn't solve the problem
# but for now let's keep doing the integration manually, it's
# clearer what's going on and performance didn't seem to take a hit
from scipy.integrate import odeint


def brute_force_r0(seir_params, new_r0, r0):
    calc_r0 = r0 * 1000
    change = np.sign(new_r0 - calc_r0) * 0.00005
    # step = 0.1
    # direction = 1 if change > 0 else -1

    new_seir_params = seir_params.copy()

    while round(new_r0, 4) != round(calc_r0, 4):
        new_seir_params["beta"] = [
            0.0,
            new_seir_params["beta"][1] + change,
            0.0,
            0.0,
        ]
        calc_r0 = generate_r0(new_seir_params) * 1000

        diff_r0 = new_r0 - calc_r0

        # if the sign has changed, we overshot, turn around with a smaller
        # step
        if np.sign(diff_r0) != np.sign(change):
            change = -change / 2

    return new_seir_params


def dataframe_ify(data, start, end, steps):
    last_period = start + datetime.timedelta(days=(steps - 1))

    timesteps = pd.date_range(
        # start=start, end=last_period, periods=steps, freq=='D',
        start=start,
        end=last_period,
        freq="D",
    ).to_list()

    sir_df = pd.DataFrame(
        zip(data[0], data[1], data[2], data[3], data[4], data[5]),
        columns=[
            "exposed",
            "infected_a",
            "infected_b",
            "infected_c",
            "recovered",
            "dead",
        ],
        index=timesteps,
    )

    # reample the values to be daily
    sir_df.resample("1D").sum()

    # drop anything after the end day
    sir_df = sir_df.loc[:end]

    return sir_df


f = """
def deriv(y0, t, beta, alpha, gamma, rho, mu, N):
    dy = [0, 0, 0, 0, 0, 0, 0]
    # S = np.max([N - sum(y0), 0])

    # print(y0[0])

    # dy[0] = max((y0[0] - (np.dot(beta[1:3], y0[2:4]) * y0[0])), 0)  # Susceptible
    dy[0] = (
        0 if y0[0] < 0 else y0[0] - (np.dot(beta[1:3], y0[2:4]) * y0[0])
    )  # Susceptible
    dy[1] = (np.dot(beta[1:3], y0[2:4]) * y0[0]) - (alpha * y0[1])  # Exposed
    dy[2] = (alpha * y0[1]) - (gamma[1] + rho[1]) * y0[2]  # Ia - Mildly ill
    dy[3] = (rho[1] * y0[2]) - (gamma[2] + rho[2]) * y0[3]  # Ib - Hospitalized
    dy[4] = (rho[2] * y0[3]) - ((gamma[3] + mu) * y0[4])  # Ic - ICU
    dy[5] = np.dot(gamma[1:3], y0[2:4])  # Recovered
    dy[6] = mu * y0[4]  # Deaths

    return dy
"""
# The SEIR model differential equations.
# https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb
# but these are the basics
# y = initial conditions
# t = a grid of time points (in days) - not currently used, but will be for time-dependent functions
# N = total pop
# beta = contact rate
# gamma = mean recovery rate
# Don't track S because all variables must add up to 1
# include blank first entry in vector for beta, gamma, p so that indices align in equations and code.
# In the future could include recovery or infection from the exposed class (asymptomatics)
def deriv(y0, t, beta, alpha, gamma, rho, mu, N):
    dy = [0, 0, 0, 0, 0, 0]
    # S = N - sum(y0)
    S = np.max([N - sum(y0), 0])

    dy[0] = np.min([(np.dot(beta[1:3], y0[1:3]) * S), S]) - (alpha * y0[0])  # Exposed
    dy[1] = (alpha * y0[0]) - (gamma[1] + rho[1]) * y0[1]  # Ia - Mildly ill
    dy[2] = (rho[1] * y0[1]) - (gamma[2] + rho[2]) * y0[2]  # Ib - Hospitalized
    dy[3] = (rho[2] * y0[2]) - ((gamma[3] + mu) * y0[3])  # Ic - ICU
    dy[4] = np.min([np.dot(gamma[1:3], y0[1:3]), sum(y0[1:3])])  # Recovered
    dy[5] = mu * y0[3]  # Deaths

    return dy


# Sets up and runs the integration
# start date and end date give the bounds of the simulation
# pop_dict contains the initial populations
# beta = contact rate
# gamma = mean recovery rate
# TODO: add other params from doc
def seir(
    pop_dict, beta, alpha, gamma, rho, mu, harvard_flag=False,
):
    if harvard_flag:
        N = 1000
        y0 = np.zeros(6)
        y0[0] = 1
        steps = 365

    else:
        N = pop_dict["total"]
        # assume that the first time you see an infected population it is mildly so
        # after that, we'll have them broken out
        if "infected_a" in pop_dict:
            first_infected = pop_dict["infected_a"]
        else:
            first_infected = pop_dict["infected"]

        susceptible = pop_dict["total"] - (
            pop_dict["infected"] + pop_dict["recovered"] + pop_dict["deaths"]
        )

        y0 = [
            # susceptible,
            pop_dict.get("exposed", 0),
            float(first_infected),
            float(pop_dict.get("infected_b", 0)),
            float(pop_dict.get("infected_c", 0)),
            float(pop_dict.get("recovered", 0)),
            float(pop_dict.get("deaths", 0)),
        ]

        steps = 365
        t = np.arange(0, steps, 1)

    # print("beta: " + str(beta[1]))

    ret = odeint(deriv, y0, t, args=(beta, alpha, gamma, rho, mu, N))

    return np.transpose(ret), steps, ret


# for testing purposes, just load the Harvard output
def harvard_model_params():
    return {
        "beta": [0.0, 0.00025, 0.0, 0.0],
        "alpha": 0.2,
        "gamma": [0.0, 0.08, 0.06818182, 0.08571429],
        "rho": [0.0, 0.02, 0.02272727],
        "mu": 0.057142857142857134,
    }


# for now just implement Harvard model, in the future use this to change
# key params due to interventions
def generate_epi_params(model_parameters):
    fraction_critical = (
        model_parameters["hospitalization_rate"]
        * model_parameters["hospitalized_cases_requiring_icu_care"]
    )

    alpha = 1 / model_parameters["total_infected_period"]

    # assume hospitalized don't infect
    beta = [0, model_parameters["r0"] / 10000, 0, 0]

    # have to calculate these in order and then put them into arrays
    gamma_0 = 0
    gamma_1 = (1 / model_parameters["duration_mild_infections"]) * (
        1 - model_parameters["hospitalization_rate"]
    )

    rho_0 = 0
    rho_1 = (1 / model_parameters["duration_mild_infections"]) - gamma_1
    rho_2 = (1 / model_parameters["hospital_time_recovery"]) * (
        fraction_critical / model_parameters["hospitalization_rate"]
    )

    gamma_2 = (1 / model_parameters["hospital_time_recovery"]) - rho_2

    mu = (1 / model_parameters["icu_time_death"]) * (
        model_parameters["case_fatality_rate"] / fraction_critical
    )
    gamma_3 = (1 / model_parameters["icu_time_death"]) - mu

    seir_params = {
        "beta": beta,
        "alpha": alpha,
        "gamma": [gamma_0, gamma_1, gamma_2, gamma_3],
        "rho": [rho_0, rho_1, rho_2],
        "mu": mu,
    }
    return seir_params


def generate_r0(seir_params):
    b = seir_params["beta"]
    p = seir_params["rho"]
    g = seir_params["gamma"]
    u = seir_params["mu"]

    r0 = (b[1] / (p[1] + g[1])) + (p[1] / (p[1] + g[1])) * (
        b[2] / (p[2] + g[2]) + (p[2] / (p[2] + g[2])) * (b[3] / (u + g[3]))
    )

    return r0
