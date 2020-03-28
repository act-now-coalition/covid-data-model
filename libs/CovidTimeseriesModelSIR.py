import logging
import math
import json

import numpy as np
import pandas as pd
import datetime
from scipy.integrate import odeint

import pprint


# The SEIRD model differential equations.
# https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb
# we'll blow these out to add E, D, and variations on I (to SIR)
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
    S = N - sum(y0)

    dy[0] = (np.dot(beta[1:3], y0[1:3]) * S) - (alpha * y0[0])  # Exposed
    dy[1] = (alpha * y0[0]) - (gamma[1] + rho[1]) * y0[1]  # Ia - Mildly ill
    dy[2] = (rho[1] * y0[1]) - (gamma[2] + rho[2]) * y0[2]  # Ib - Hospitalized
    dy[3] = (rho[2] * y0[2]) - ((gamma[3] + mu) * y0[3])  # Ic - ICU
    dy[4] = np.dot(gamma[1:3], y0[1:3])  # Recovered
    dy[5] = mu * y0[3]  # Deaths

    return dy


# @TODO: Switch to this model.
class CovidTimeseriesCycle(object):
    def __init__(self, model_parameters, init_data_cycle):
        # There are some assumptions made when we create the fisrt data cycle. This encapsulates all of those
        #  assumptions into one place

        self.date = model_parameters["init_date"]
        self.r0 = model_parameters["r0"]
        self.effective_r0 = None
        self.cases = init_data_cycle["cases"]
        self.actual_reported = init_data_cycle["cases"]
        self.current_infected = 0
        self.newly_infected_from_confirmed = (
            init_data_cycle["cases"]
            * model_parameters["estimated_new_cases_per_confirmed"]
        )
        self.newly_infected_from_deaths = 0
        self.newly_infected = (
            self.newly_infected_from_confirmed + self.newly_infected_from_deaths
        )
        self.currently_infected = 0
        self.cumulative_infected = None
        self.cumulative_deaths = None
        self.recovered_or_died = 0
        self.ending_susceptible = model_parameters["population"] - (
            init_data_cycle["cases"] / model_parameters["hospitalization_rate"]
        )
        self.predicted_hospitalized = 0
        self.available_hospital_beds = model_parameters[
            "original_available_hospital_beds"
        ]

    def __iter__(self):
        return self

    def __next__(self):
        pass


class CovidTimeseriesModelSIR:
    # Initializer / Instance Attributes
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)

    def calculate_r(self, current_cycle, previous_cycle, model_parameters):
        # Calculate the r0 value based on the current and past number of confirmed cases
        if current_cycle["cases"] is not None:
            if previous_cycle["cases"] > 0:
                return current_cycle["cases"] / previous_cycle["cases"]
        return model_parameters["r0"]

    def calculate_effective_r(self, current_cycle, previous_cycle, model_parameters):
        # In order to account for interventions, we need to allow effective r to be changed manually.
        if model_parameters["interventions"] is not None:
            # The interventions come in as a dict, with date's as keys and effective r's as values
            # Find the most recent intervention we have passed
            for d in sorted(model_parameters["interventions"].keys())[::-1]:
                if (
                    current_cycle["date"] >= d
                ):  # If the current cycle occurs on or after an intervention
                    return model_parameters["interventions"][d] * (
                        previous_cycle["ending_susceptible"]
                        / model_parameters["population"]
                    )
        # Calculate effective r by accounting for herd-immunity
        return current_cycle["r"] * (
            previous_cycle["ending_susceptible"] / model_parameters["population"]
        )

    def calculate_newly_infected(self, current_cycle, previous_cycle, model_parameters):
        # If we have previously known cases, use the R0 to estimate newly infected cases.
        nic = 0
        if previous_cycle["newly_infected"] > 0:
            nic = previous_cycle["newly_infected"] * current_cycle["effective_r"]
        elif current_cycle["cases"] is not None:
            if not "newly_infected" in current_cycle:
                pprint.pprint(current_cycle)
            nic = (
                current_cycle["newly_infected"]
                * model_parameters["estimated_new_cases_per_confirmed"]
            )

        nid = 0
        if current_cycle["deaths"] is not None:
            nid = (
                current_cycle["deaths"]
                * model_parameters["estimated_new_cases_per_death"]
            )

        return (nic, nid)

    def calculate_currently_infected(
        self, cycle_series, rolling_intervals_for_current_infected
    ):
        # Calculate the number of people who have been infected but are no longer infected (one way or another)
        # Better way:
        return sum(
            ss["newly_infected"]
            for ss in cycle_series[-rolling_intervals_for_current_infected:]
        )

    def hospital_is_overloaded(self, current_cycle, previous_cycle):
        # Determine if a hospital is overloaded. Trivial now, but will likely become more complicated in the near future
        return (
            previous_cycle["available_hospital_beds"]
            < current_cycle["predicted_hospitalized"]
        )

    def calculate_cumulative_infected(self, current_cycle, previous_cycle):
        if previous_cycle["cumulative_infected"] is None:
            return current_cycle["newly_infected"]
        return previous_cycle["cumulative_infected"] + current_cycle["newly_infected"]

    def calculate_cumulative_deaths(
        self,
        current_cycle,
        previous_cycle,
        case_fatality_rate,
        case_fatality_rate_hospitals_overwhelmed,
    ):
        # If the number of hospital beds available is exceeded by the number of patients that need hospitalization,
        #  the death rate increases
        # Can be significantly improved in the future
        if previous_cycle["cumulative_deaths"] is None:
            return current_cycle["cumulative_infected"] * case_fatality_rate
        if not self.hospital_is_overloaded(current_cycle, previous_cycle):
            return previous_cycle["cumulative_deaths"] + (
                current_cycle["newly_infected"] * case_fatality_rate
            )
        else:
            return previous_cycle["cumulative_deaths"] + (
                current_cycle["newly_infected"]
                * (case_fatality_rate + case_fatality_rate_hospitals_overwhelmed)
            )

    def calculuate_recovered_or_died(
        self,
        current_cycle,
        previous_cycle,
        cycle_series,
        rolling_intervals_for_current_infected,
    ):
        # Recovered or died (RoD) is a cumulative number. We take the number of RoD from last current_cycle, and add to it
        #  the number of individuals who were newly infected a set number of current_cycles ago (rolling_intervals_for_current_infected)
        #  (RICI). It is assumed that after the RICI, an infected interval has either recovered or died.
        # Can be improved in the future
        if len(cycle_series) >= (rolling_intervals_for_current_infected + 1):
            return (
                previous_cycle["recovered_or_died"]
                + cycle_series[-(rolling_intervals_for_current_infected + 1)][
                    "newly_infected"
                ]
            )
        else:
            return previous_cycle["recovered_or_died"]

    def calculate_predicted_hospitalized(self, current_cycle, model_parameters):
        return (
            current_cycle["newly_infected"] * model_parameters["hospitalization_rate"]
        )

    def calculate_ending_susceptible(
        self, current_cycle, previous_cycle, model_parameters
    ):
        # Number of people who haven't been sick yet
        return model_parameters["population"] - (
            current_cycle["newly_infected"]
            + current_cycle["currently_infected"]
            + current_cycle["recovered_or_died"]
        )

    def calculate_estimated_actual_chance_of_infection(
        self, current_cycle, previous_cycle, hospitalization_rate, pop
    ):
        # Reflects the model logic, but probably needs to be changed
        if current_cycle["cases"] is not None:
            return ((current_cycle["cases"] / hospitalization_rate) * 2) / pop
        return None

    def calculate_actual_reported(self, current_cycle, previous_cycle):
        # Needs to account for creating synthetic data for missing records
        if current_cycle["cases"] is not None:
            return current_cycle["cases"]
        return None

    def calculate_available_hospital_beds(
        self,
        current_cycle,
        previous_cycle,
        max_hospital_capacity_factor,
        original_available_hospital_beds,
        hospital_capacity_change_daily_rate,
    ):
        available_hospital_beds = previous_cycle["available_hospital_beds"]
        if current_cycle["i"] < 3:
            return available_hospital_beds
        if (
            available_hospital_beds
            < max_hospital_capacity_factor * original_available_hospital_beds
        ):
            # Hospitals can try to increase their capacity for the sick
            available_hospital_beds *= hospital_capacity_change_daily_rate
        return available_hospital_beds

    def run_interventions(self, interventions, combined_df, seird_params):
        ## for each intervention (in order)
        ## grab initial conditions (conditions at intervention date)
        ## adjust seird_params based on intervention
        ## run model from that date with initial conditions and new params
        ## merge new dataframes, keep old one as counterfactual for that intervention
        ## rinse, repeat

        return post_interventions_df, counterfactuals

    def initialize_parameters(self, model_parameters):
        """Perform all of the necessary setup prior to the model's execution"""
        # want first and last days from the actual values in timeseries
        actual_values = model_parameters["timeseries"].sort_values("date")

        model_parameters["actual_init_date"] = actual_values.iloc[
            0, actual_values.columns.get_loc("date")
        ]

        model_parameters["actual_end_date"] = actual_values.iloc[
            -1, actual_values.columns.get_loc("date")
        ]

        # TODO: add check for earlier int date parameter and adjust so that
        # we can start the run earlier than the last data point

        model_parameters["init_date"] = model_parameters["actual_end_date"]

        # Get the last day of the model based on the number of iterations and
        # length of the iterations
        duration = datetime.timedelta(
            days=(
                model_parameters["model_interval"]
                * model_parameters["projection_iterations"]
            )
        )
        model_parameters["last_date"] = model_parameters["init_date"] + duration

        return model_parameters

    # for testing purposes, just load the Harvard output
    def harvard_model_params(self):
        return {
            "beta": [0.0, 0.00025, 0.0, 0.0],
            "alpha": 0.2,
            "gamma": [0.0, 0.08, 0.06818182, 0.08571429],
            "rho": [0.0, 0.02, 0.02272727],
            "mu": 0.057142857142857134,
        }

    # for now just implement Harvard model, in the future use this to change
    # key params due to interventions
    def generate_seird_params(self, model_parameters):
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

        seird_params = {
            "beta": beta,
            "alpha": alpha,
            "gamma": [gamma_0, gamma_1, gamma_2, gamma_3],
            "rho": [rho_0, rho_1, rho_2],
            "mu": mu,
        }
        return seird_params

    # Sets up and runs the integration
    # start date and end date give the bounds of the simulation
    # pop_dict contains the initial populations
    # beta = contact rate
    # gamma = mean recovery rate
    # TODO: add other params from doc
    def seird(
        self, start_date, end_date, pop_dict, beta, alpha, gamma, rho, mu, harvard_flag
    ):
        if harvard_flag:
            N = 1000
            y0 = np.zeros(6)
            y0[0] = 1
            steps = 365
            t = np.arange(0, steps, 0.1)
            steps = steps * 10
        else:
            N = pop_dict["total"]
            # assume that the first time you see an infected population it is mildly so
            # after that, we'll have them broken out
            if "infected_a" in pop_dict:
                first_infected = pop_dict["infected_a"]
            else:
                first_infected = pop_dict["infected"]

            y0 = np.array(
                (
                    pop_dict.get("exposed", 0),
                    float(first_infected),
                    float(pop_dict.get("infected_b", 0)),
                    float(pop_dict.get("infected_c", 0)),
                    float(pop_dict["recovered"]),
                    float(pop_dict["deaths"]),
                )
            )

            steps = 365
            t = np.arange(0, steps, 1)

        print(y0)
        print(beta, alpha, gamma, rho, mu, N)
        ret = odeint(deriv, y0, t, args=(beta, alpha, gamma, rho, mu, N))

        return ret.T, steps, ret

    def dataframe_ify(self, data, start, end, steps):
        last_period = start + datetime.timedelta(days=(steps / 10))

        timesteps = pd.date_range(
            start=start, end=last_period, periods=steps, freq=None,
        ).to_list()

        sir_df = pd.DataFrame(
            zip(data[0], data[1], data[2], data[3], data[4], data[5]),
            columns=[
                "exposed",  # "susceptible"
                "infected_a",
                "infected_b",
                "infected_c",
                "recovered",
                "dead",
            ],
            index=timesteps,
        )

        # reample the values to be daily
        # drop anything after the end day
        sir_df.resample("1D").sum().loc[:end]

        return sir_df

    def iterate_model(self, model_parameters):
        """The guts. Creates the initial conditions, and runs the SIR model for the
        specified number of iterations with the given inputs"""

        ## TODO:
        ## implement interventions
        #
        ## pull together interventions into the date they take place
        #
        ## nice-to have - counterfactuals for interventions

        timeseries = model_parameters["timeseries"].sort_values("date")

        # calc values if missing
        timeseries.loc[:, ["cases", "deaths", "recovered"]] = timeseries.loc[
            :, ["cases", "deaths", "recovered"]
        ].fillna(0)

        timeseries["active"] = (
            timeseries["cases"] - timeseries["deaths"] - timeseries["recovered"]
        )

        # timeseries["active"] = timeseries["active"].fillna(timeseries["active_calc"])

        model_parameters["timeseries"] = timeseries

        model_parameters = self.initialize_parameters(model_parameters)

        timeseries["dt"] = pd.to_datetime(timeseries["date"]).dt.date
        timeseries.set_index("dt", inplace=True)
        timeseries.sort_index(inplace=True)

        init_date = model_parameters["init_date"].to_pydatetime().date()

        print(timeseries.tail())

        # load the initial populations
        pop_dict = {
            "total": model_parameters["population"],
            "infected": timeseries.loc[init_date, "active"],
            "recovered": timeseries.loc[init_date, "recovered"],
            "deaths": timeseries.loc[init_date, "deaths"],
        }

        if model_parameters["use_harvard_params"]:
            init_params = self.harvard_model_params()
        else:
            init_params = self.generate_seird_params(model_parameters)

        (data, steps, ret) = self.seird(
            model_parameters["init_date"],
            model_parameters["last_date"],
            pop_dict,
            init_params["beta"],
            init_params["alpha"],
            init_params["gamma"],
            init_params["rho"],
            init_params["mu"],
            model_parameters["use_harvard_init"],
        )

        # this dataframe should start on the last day of the actual data
        # and have the same values for those initial days, so we combine it with
        # the slice of timeseries from the actual_init_date to actual_end_date - 1
        sir_df = self.dataframe_ify(
            data, model_parameters["init_date"], model_parameters["last_date"], steps,
        )

        sir_df["infected"] = (
            sir_df["infected_a"] + sir_df["infected_b"] + sir_df["infected_c"]
        )

        if model_parameters["use_harvard_init"]:
            combined_df = sir_df
            sir_df["total"] = 1000

            sir_df["susceptible"] = sir_df.total - (
                sir_df.exposed + sir_df.infected + sir_df.recovered + sir_df.dead
            )
        else:
            sir_df["total"] = pop_dict["total"]

            sir_df["susceptible"] = sir_df.total - (
                sir_df.exposed + sir_df.infected + sir_df.recovered + sir_df.dead
            )

            timeseries["susceptible"] = model_parameters["population"] - (
                timeseries.active + timeseries.recovered + timeseries.deaths
            )

            actual_cols = ["population", "susceptible", "active", "recovered", "deaths"]
            # kill last row that is initial conditions on SEIRD
            actuals = timeseries.loc[:, actual_cols].head(-1)

            # it wasn't a df thing, you can rip all this out
            actuals.rename(
                columns={"population": "total", "deaths": "dead", "active": "infected"},
                inplace=True,
            )

            actuals.index = pd.to_datetime(actuals.index, format="%Y-%m-%d")

            actuals["infected_a"] = 0
            actuals["infected_b"] = 0
            actuals["infected_c"] = 0

            all_cols = [
                "total",
                "susceptible",
                "exposed",
                "infected",
                "infected_a",
                "infected_b",
                "infected_c",
                "recovered",
                "dead",
            ]

            actuals = actuals.loc[:, all_cols]
            sir_df = sir_df.loc[:, all_cols]

            combined_df = pd.concat([actuals, sir_df])

            # this should be done, but belt and suspenders for the diffs()
            combined_df.sort_index(inplace=True)
            combined_df.index.name = "date"
            combined_df.reset_index(inplace=True)

            combined_df["total"] = pop_dict["total"]

            # move the actual infected numbers into infected_a where its NA
            combined_df["infected_a"] = combined_df["infected_a"].fillna(
                combined_df["infected"]
            )

            if model_parameters["interventions"] is not None:
                combo_df, counterfactuals = self.run_interventions(
                    model_parameters["interventions"], combined_df, init_params
                )

        # set some of the paramters... I'm sure I'm misinterpreting some
        # and of course a lot of these don't move like they should for the model yet
        # combined_df["r"] = model_parameters["r0"]
        # combined_df["effective_r"] = model_parameters["r0"]
        # combined_df["ending_susceptible"] = combined_df["susceptible"]
        # combined_df["currently_infected"] = combined_df["infected"]

        # not sure about these guys... just doing new infections
        # combined_df["newly_infected_from_confirmed"] = combined_df["infected"].diff()
        # combined_df["newly_infected_from_deaths"] = combined_df["infected"].diff()

        # fillna
        # combined_df.loc[
        #    :, ["newly_infected_from_confirmed", "newly_infected_from_deaths"]
        # ] = combined_df.loc[
        #    :, ["newly_infected_from_confirmed", "newly_infected_from_deaths"]
        # ].fillna(
        #    0
        # )

        # cumsum the diff (no D yet)
        # combined_df["cumulative_infected"] = combined_df[
        #    "newly_infected_from_confirmed"
        # ].cumsum()

        # no D yet in model
        # combined_df["recovered_or_died"] = combined_df["recovered"]

        # cumsum the diff (no D yet)
        # combined_df["newly_died"] = combined_df["recovered"].diff()
        # combined_df["cumulative_deaths"] = combined_df["newly_died"].cumsum()

        # have not broken out asymptomatic from hospitalized/severe yet
        # combined_df["predicted_hospitalized"] = combined_df["infected"]

        # combined_df["newly_infected"] = combined_df["infected"]

        # TODO: work on all these guys
        # 'actual_reported'
        # 'predicted_hospitalized'
        # 'cumulative_infected'
        # 'cumulative_deaths'
        # 'available_hospital_beds'
        # combined_df["actual_reported"] = 0
        # combined_df["predicted_hospitalized"] = 0
        # combined_df["available_hospital_beds"] = 0

        # return combined_df.to_dict("records")  # cycle_series
        return [combined_df, ret]
        # return ret

    def forecast_region(self, model_parameters):
        cycle_series = self.iterate_model(model_parameters)

        return cycle_series

        tmp = """
        return pd.DataFrame(
            {
                "Date": [s["date"] for s in cycle_series],
                "Timestamp": [
                    # Create a UNIX timestamp for each datetime. Easier for graphs to digest down the road
                    datetime.datetime(
                        year=s["date"].year, month=s["date"].month, day=s["date"].day
                    ).timestamp()
                    for s in cycle_series
                ],
                "R": [s["r"] for s in cycle_series],
                "Effective R.": [s["effective_r"] for s in cycle_series],
                "Beg. Susceptible": [s["ending_susceptible"] for s in cycle_series],
                "New Inf < C": [
                    int(round(s["newly_infected_from_confirmed"])) for s in cycle_series
                ],
                "New Inf < D": [
                    int(round(s["newly_infected_from_deaths"])) for s in cycle_series
                ],
                "New Inf.": [int(round(s["newly_infected"])) for s in cycle_series],
                "Curr. Inf.": [s["currently_infected"] for s in cycle_series],
                "Recov. or Died": [s["recovered_or_died"] for s in cycle_series],
                "End Susceptible": [s["ending_susceptible"] for s in cycle_series],
                "Actual Reported": [s["actual_reported"] for s in cycle_series],
                "Pred. Hosp.": [s["predicted_hospitalized"] for s in cycle_series],
                "Cum. Inf.": [s["cumulative_infected"] for s in cycle_series],
                "Cum. Deaths": [s["cumulative_deaths"] for s in cycle_series],
                "Avail. Hosp. Beds": [
                    s["available_hospital_beds"] for s in cycle_series
                ],
            }
        )
        """
