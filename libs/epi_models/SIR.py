import numpy as np
import pandas as pd

# odeint might work, moving it out didn't solve the problem
# but for now let's keep doing the integration manually, it's
# clearer what's going on and performance didn't seem to take a hit
from scipy.integrate import odeint
import datetime


def dataframe_ify(data, start, end, steps):
    last_period = start + datetime.timedelta(days=(steps - 1))

    timesteps = pd.date_range(
        # start=start, end=last_period, periods=steps, freq=='D',
        start=start,
        end=last_period,
        freq="D",
    ).to_list()

    sir_df = pd.DataFrame(
        # zip(data[0], data[1], data[2], data[3], data[4], data[5], data[6]),
        zip(data[0], data[1], data[2]),
        columns=["susceptible", "infected", "recovered",],
        index=timesteps,
    )

    # reample the values to be daily
    sir_df.resample("1D").sum()

    # drop anything after the end day
    sir_df = sir_df.loc[:end]

    # calculate dead
    sir_df["dead"] = sir_df["recovered"] * 0.008
    # reomve from recovered
    sir_df["recovered"] = sir_df["recovered"] - sir_df["dead"]

    sir_df["infected_a"] = 0
    sir_df["infected_b"] = 0
    sir_df["infected_c"] = 0

    sir_df["exposed"] = 0

    return sir_df


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
def deriv(y0, t, beta, gamma, N):
    dy = [0, 0, 0]

    S = y0[0]

    beta, gamma = 0.2, 1.0 / 10

    dy[0] = -beta * S * y0[1] / N  # Susceptible
    dy[1] = beta * S * y0[1] / N - gamma * y0[1]  # Infected
    dy[2] = gamma * y0[1]  # Recovered

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
        susceptible,
        float(first_infected),
        float(pop_dict.get("recovered", 0)),
    ]

    steps = 365
    t = np.arange(0, steps, 1)

    ret = odeint(deriv, y0, t, args=(beta, gamma, N))
    return np.transpose(ret), steps, ret


# for now just implement Harvard model, in the future use this to change
# key params due to interventions
def generate_epi_params(model_parameters):
    fraction_critical = (
        model_parameters["hospitalization_rate"]
        * model_parameters["hospitalized_cases_requiring_icu_care"]
    )

    # assume hospitalized don't infect
    # TODO make a real beta
    beta = model_parameters["r0"] / 10000

    gamma = 1 / model_parameters["hospital_time_recovery"]

    seir_params = {
        "beta": beta,
        "alpha": 0,
        "gamma": gamma,
        "rho": 0,
        # TODO: add this parameter
        # "mu": model_parameters["sir_death_rate"],
        "mu": 0.008,
    }
    return seir_params


def generate_r0(seir_params):
    r0 = seir_params["beta"] / seir_params["gamma"]

    return r0


def brute_force_r0(seir_params, new_r0, r0):
    calc_r0 = r0 * 1000
    change = np.sign(new_r0 - calc_r0) * 0.00005
    # step = 0.1
    # direction = 1 if change > 0 else -1

    new_seir_params = seir_params.copy()

    while round(new_r0, 4) != round(calc_r0, 4):
        new_seir_params["beta"] = new_seir_params["beta"] + change

        calc_r0 = generate_r0(new_seir_params) * 1000

        diff_r0 = new_r0 - calc_r0

        # if the sign has changed, we overshot, turn around with a smaller
        # step
        if np.sign(diff_r0) != np.sign(change):
            change = -change / 2

    return new_seir_params
