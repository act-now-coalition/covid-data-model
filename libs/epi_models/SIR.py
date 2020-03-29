import numpy as np

# odeint might work, moving it out didn't solve the problem
# but for now let's keep doing the integration manually, it's
# clearer what's going on and performance didn't seem to take a hit
from scipy.integrate import odeint


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
    dy = [0, 0, 0]

    S = y0[0]

    dy[0] = -beta[0] * S * y0[1] / N  # Susceptible
    dy[1] = beta[0] * S * y0[1] / N - gamma[0] * y0[1]  # Infected
    dy[2] = gamma[0] * y0[1]  # Recovered

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

    y = np.zeros((3, steps))
    y[:, 0] = y0

    for day in range(steps - 1):
        y[:, day + 1] = y[:, day] + deriv(y[:, day], 1, beta, alpha, gamma, rho, mu, N)
        # y[5, day + 1] = y[2, day + 1] * 0.008
        # y[2, day + 1] = y[2, day + 1] - y[5, day + 1]
        print(y[:, day + 1])
    # for reasons that are beyond me the odeint doesn't work
    # ret = odeint(deriv, y0, t, args=(beta, alpha, gamma, rho, mu, N))

    return y, steps, np.transpose(y)
