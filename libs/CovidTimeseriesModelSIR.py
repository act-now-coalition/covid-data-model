import logging
import math
import json

import numpy as np
import pandas as pd
import datetime

import pprint

from .epi_models.HarvardEpi import seir


class CovidTimeseriesModelSIR:
    # Initializer / Instance Attributes
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)

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

    def gen_r0(self, seir_params):
        b = seir_params["beta"]
        p = seir_params["rho"]
        g = seir_params["gamma"]
        u = seir_params["mu"]

        r0 = (b[1] / (p[1] + g[1])) + (p[1] / (p[1] + g[1])) * (
            b[2] / (p[2] + g[2]) + (p[2] / (p[2] + g[2])) * (b[3] / (u + g[3]))
        )

        return r0

    # for now just implement Harvard model, in the future use this to change
    # key params due to interventions
    def generate_seir_params(self, model_parameters):
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

    def dataframe_ify(self, data, start, end, steps):
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

    def brute_force_r0(self, seir_params, new_r0, r0):
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
            calc_r0 = self.gen_r0(new_seir_params) * 1000

            diff_r0 = new_r0 - calc_r0

            # if the sign has changed, we overshot, turn around with a smaller
            # step
            if np.sign(diff_r0) != np.sign(change):
                change = -change / 2

        return new_seir_params

    def run_interventions(self, model_parameters, combined_df, seir_params, r0):
        ## for each intervention (in order)
        ## grab initial conditions (conditions at intervention date)
        ## adjust seir_params based on intervention
        ## run model from that date with initial conditions and new params
        ## merge new dataframes, keep old one as counterfactual for that intervention
        ## rinse, repeat
        interventions = model_parameters["interventions"]
        end_date = model_parameters["last_date"]

        counterfactuals = {}

        for date, new_r0 in interventions.items():
            if (pd.Timestamp(date) >= model_parameters["init_date"]) and (
                pd.Timestamp(date) <= end_date
            ):

                counterfactuals[date] = combined_df

                new_seir_params = self.brute_force_r0(seir_params, new_r0, r0)

                # this is a dumb way to do this, but it might work
                combined_df.loc[:, "infected"] = (
                    combined_df.loc[:, "infected_a"]
                    + combined_df.loc[:, "infected_b"]
                    + combined_df.loc[:, "infected_c"]
                )

                pop_dict = {
                    "total": model_parameters["population"],
                    "infected": combined_df.loc[date, "infected"],
                    "infected_a": combined_df.loc[date, "infected_a"],
                    "infected_b": combined_df.loc[date, "infected_a"],
                    "infected_c": combined_df.loc[date, "infected_a"],
                    "recovered": combined_df.loc[date, "recovered"],
                    "deaths": combined_df.loc[date, "dead"],
                }

                (data, steps, ret) = seir(
                    pop_dict,
                    new_seir_params["beta"],
                    new_seir_params["alpha"],
                    new_seir_params["gamma"],
                    new_seir_params["rho"],
                    new_seir_params["mu"],
                    False,
                )

                new_df = self.dataframe_ify(data, date, end_date, steps,)

                early_combo_df = combined_df.copy().loc[:date]

                combined_df = early_combo_df.append(new_df, sort=True)

        return (combined_df, counterfactuals)

    def iterate_model(self, model_parameters):
        """The guts. Creates the initial conditions, and runs the SIR model for the
        specified number of iterations with the given inputs"""

        ## TODO:
        ## implement interventions
        #
        ## pull together interventions into the date they take place
        #
        ## nice-to have - counterfactuals for interventions

        # hack for total population
        model_parameters["population"] = 10000

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

        # load the initial populations
        pop_dict = {
            "total": model_parameters["population"],
            # "total": 10000,  # model_parameters["population"],
            "infected": timeseries.loc[init_date, "active"],
            "recovered": timeseries.loc[init_date, "recovered"],
            "deaths": timeseries.loc[init_date, "deaths"],
        }

        if model_parameters["use_harvard_params"]:
            init_params = self.harvard_model_params()
        else:
            init_params = self.generate_seir_params(model_parameters)

        r0 = self.gen_r0(init_params)

        (data, steps, ret) = seir(
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

            timeseries["susceptible"] = model_parameters["population"] - (
                timeseries.active + timeseries.recovered + timeseries.deaths
            )

            actual_cols = ["population", "susceptible", "active", "recovered", "deaths"]
            # kill last row that is initial conditions on SEIR
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

            actuals["exposed"] = 0

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

            if model_parameters["interventions"] is not None:
                (combined_df, counterfactuals) = self.run_interventions(
                    model_parameters, combined_df, init_params, r0
                )

                print(combined_df.tail(1))

            # this should be done, but belt and suspenders for the diffs()
            combined_df.sort_index(inplace=True)
            combined_df.index.name = "date"
            combined_df.reset_index(inplace=True)

            combined_df["total"] = pop_dict["total"]

            # move the actual infected numbers into infected_a where its NA
            combined_df["infected_a"] = combined_df["infected_a"].fillna(
                combined_df["infected"]
            )

            # make infected total represent the sum of the infected stocks
            combined_df.loc[:, "infected"] = (
                combined_df.loc[:, "infected_a"]
                + combined_df.loc[:, "infected_b"]
                + combined_df.loc[:, "infected_c"]
            )

            combined_df["susceptible"] = combined_df.total - (
                combined_df.exposed
                + combined_df.infected
                + combined_df.recovered
                + combined_df.dead
            )

        return [combined_df, ret]

    def forecast_region(self, model_parameters):
        cycle_series = self.iterate_model(model_parameters)

        return cycle_series
