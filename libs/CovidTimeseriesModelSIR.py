import logging
import math
import json
import datetime


import numpy as np
import pandas as pd

# from .epi_models.HarvardEpi import (
#    seir,
#    dataframe_ify,
#    generate_epi_params,
#    generate_r0,
#    brute_force_r0,
# )
from .epi_models.SIR import (
    seir,
    dataframe_ify,
    generate_epi_params,
    generate_r0,
    brute_force_r0,
)


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

    # get the largest key that is less than the intervention date and reurn the relevant r0
    def get_latest_past_intervention(self, interventions, init_date):
        past_dates = [
            interevention_date
            for interevention_date in interventions.keys()
            if interevention_date <= init_date
        ]

        if len(past_dates) > 0:
            return interventions[max(past_dates)]
        else:
            return None

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

                new_seir_params = brute_force_r0(seir_params, new_r0, r0)

                pop_dict = {
                    "total": model_parameters["population"],
                    "infected": combined_df.loc[date, "infected"],
                    "recovered": combined_df.loc[date, "recovered"],
                    "deaths": combined_df.loc[date, "dead"],
                }

                if model_parameters["interventions"] == "seir":
                    # this is a dumb way to do this, but it might work
                    combined_df.loc[:, "infected"] = (
                        combined_df.loc[:, "infected_a"]
                        + combined_df.loc[:, "infected_b"]
                        + combined_df.loc[:, "infected_c"]
                    )

                    pop_dict["infected_a"] = combined_df.loc[date, "infected_a"]
                    pop_dict["infected_b"] = combined_df.loc[date, "infected_b"]
                    pop_dict["infected_c"] = combined_df.loc[date, "infected_c"]

                (data, steps, ret) = seir(
                    pop_dict,
                    new_seir_params["beta"],
                    new_seir_params["alpha"],
                    new_seir_params["gamma"],
                    new_seir_params["rho"],
                    new_seir_params["mu"],
                    False,
                )

                new_df = dataframe_ify(data, date, end_date, steps,)

                early_combo_df = combined_df.copy().loc[:date]

                combined_df = early_combo_df.append(new_df, sort=True)

        return (combined_df, counterfactuals)

    def iterate_model(self, model_parameters):
        """The guts. Creates the initial conditions, and runs the SIR model for the
        specified number of iterations with the given inputs"""

        ## TODO: nice-to have - counterfactuals for interventions

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
            "infected": timeseries.loc[init_date, "active"],
            "recovered": timeseries.loc[init_date, "recovered"],
            "deaths": timeseries.loc[init_date, "deaths"],
        }

        if model_parameters["use_harvard_params"]:
            init_params = harvard_model_params()
        else:
            init_params = generate_epi_params(model_parameters)

            if model_parameters["interventions"] is not None:
                new_r0 = self.get_latest_past_intervention(
                    model_parameters["interventions"], init_date
                )

                if new_r0 is not None:
                    init_params = brute_force_r0(
                        init_params, new_r0, generate_r0(init_params)
                    )

        r0 = generate_r0(init_params)

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
        sir_df = dataframe_ify(
            data, model_parameters["init_date"], model_parameters["last_date"], steps,
        )

        if model_parameters["interventions"] == "seir":
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

            actual_cols = ["active", "recovered", "deaths"]

            # kill last row that is initial conditions on SEIR
            actuals = timeseries.loc[:, actual_cols].head(-1)

            actuals["population"] = model_parameters["population"]

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

            actuals["susceptible"] = 0
            sir_df["susceptible"] = 0

            actuals = actuals.loc[:, all_cols]
            sir_df = sir_df.loc[:, all_cols]

            combined_df = pd.concat([actuals, sir_df])

            if model_parameters["interventions"] is not None:
                (combined_df, counterfactuals) = self.run_interventions(
                    model_parameters, combined_df, init_params, r0
                )

            # this should be done, but belt and suspenders for the diffs()
            combined_df.sort_index(inplace=True)
            combined_df.index.name = "date"
            combined_df.reset_index(inplace=True)

            combined_df["total"] = pop_dict["total"]

            # move the actual infected numbers into infected_a where its NA
            combined_df["infected_a"] = combined_df["infected_a"].fillna(
                combined_df["infected"]
            )

            if model_parameters["interventions"] == "seir":

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
