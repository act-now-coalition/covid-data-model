import datetime
import logging
import math
from copy import copy

import pandas as pd


class CovidDatasets:

    ## Constants
    POPULATION_URL = "https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/populations.csv"
    BED_URL = "https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/beds.csv"
    TIME_SERIES_URL = "https://coronadatascraper.com/timeseries.csv"
    DATE_FIELD = "date"
    TIME_SERIES_DATA = None
    BED_DATA = None
    POPULATION_DATA = None
    start_date = datetime.datetime(year=2020, month=3, day=3)
    # Initializer / Instance Attributes
    def __init__(self, filter_past_date=None):
        self.filter_past_date = pd.Timestamp(filter_past_date)
        logging.basicConfig(level=logging.CRITICAL)

    def get_all_countries(self):
        return self.get_all_population()["country"].unique()

    def get_all_states_by_country(self, country):
        return (
            self.get_all_population()[self.get_all_population()["country"] == country][
                "state"
            ]
            .dropna()
            .unique()
        )

    def backfill_to_init_date(self, series, model_interval):
        # We need to make sure that the data starts from Mar3, no matter when our records begin
        series = series.sort_values(self.DATE_FIELD).reset_index()
        data_rows = series[series["cases"] > 0]
        interval_rows = data_rows[
            data_rows["date"].apply(
                lambda d: (d - self.start_date).days % model_interval == 0
            )
        ]
        min_interval_row = interval_rows[
            interval_rows["date"] == interval_rows["date"].min()
        ].iloc[0]
        series = series[series["date"] >= min_interval_row["date"]]

        series["synthetic"] = None
        init_data_date = series[self.DATE_FIELD].min()
        synthetic_interval = (
            init_data_date - self.start_date
        ).days  # The number of days we need to create to backfill to Mar3
        template = series.iloc[0]  # grab a row for us to copy structure and data from
        synthetic_data = []
        pd.set_option(
            "mode.chained_assignment", None
        )  # Suppress pandas' anxiety. We know what we're doing
        for i in range(0, synthetic_interval):
            synthetic_row = template
            synthetic_row[self.DATE_FIELD] = self.start_date + datetime.timedelta(
                days=i
            )
            synthetic_row["cases"] = 0
            synthetic_row["deaths"] = 0
            synthetic_row["recovered"] = 0
            synthetic_row["active"] = 0
            synthetic_row["synthetic"] = 1
            synthetic_data.append(
                copy(synthetic_row)
            )  # We need to copy it to prevent alteration by reference
        pd.set_option("mode.chained_assignment", "warn")  # Turn the anxiety back on
        synthetic_series = pd.DataFrame(synthetic_data)
        new_series = series.append(synthetic_series)
        return new_series.sort_values("date").reset_index()

    def step_down(self, i, series, model_interval):
        # A function to calculate how much to step down the number of cases from the following day
        #  The goal is for the synthetic cases to halve once every iteration of the model interval.
        #
        # interval_rows = data_rows[data_rows['date'].apply(lambda d: (d - self.start_date).days % model_interval == 0)]
        # min_interval_row = interval_rows[interval_rows['date'] == interval_rows['date'].min()].iloc[0]

        min_row = series[series["cases"] > 0].min()
        y = min_row["cases"] / (math.pow(2, (1 / model_interval)))
        return y

    def backfill_synthetic_cases(self, series, model_interval):
        # Fill in all values prior to the first non-zero values. Use 1/2 following value. Decays into nothing
        #  sort the dataframe in reverse date order, so we traverse from latest to earliest
        for a in range(0, len(series)):
            i = len(series) - a - 1
            if series.iloc[i]["cases"] == 0:
                series.at[i, "cases"] = self.step_down(i, series, model_interval)
        return series

    def backfill(self, series, model_interval):
        # Backfill the data as necessary for the model
        return self.backfill_synthetic_cases(
            self.backfill_to_init_date(series, model_interval), model_interval
        )

    def cutoff(self, series):
        # The model MUST start on a certain day. If there is data that precedes that date,
        #  we must trim it from the series
        return series[series[self.DATE_FIELD] >= self.start_date]

    def prep_data(self, series, model_interval):
        # We have some requirements of the data's window, and that is enforced here.
        return self.cutoff(self.backfill(series, model_interval))

    def get_all_timeseries(self):
        if self.TIME_SERIES_DATA is None:
            self.TIME_SERIES_DATA = pd.read_csv(self.TIME_SERIES_URL)
            self.TIME_SERIES_DATA[self.DATE_FIELD] = pd.to_datetime(
                self.TIME_SERIES_DATA[self.DATE_FIELD]
            )
            if self.filter_past_date is not None:
                self.TIME_SERIES_DATA = self.TIME_SERIES_DATA[
                    self.TIME_SERIES_DATA[self.DATE_FIELD] <= self.filter_past_date
                ]
        return self.TIME_SERIES_DATA

    def get_all_population(self):
        if self.POPULATION_DATA is None:
            self.POPULATION_DATA = pd.read_csv(self.POPULATION_URL)
        return self.POPULATION_DATA

    def get_all_beds(self):
        if self.BED_DATA is None:
            self.BED_DATA = pd.read_csv(self.BED_URL)
        return self.BED_DATA

    def combine_state_county_data(self, country, state):
        # Create a single dataset from state and county data, using state data preferentially.
        # First, pull all available state data
        state_data = self.get_all_timeseries()[
            (self.get_all_timeseries()["state"] == state)
            & (self.get_all_timeseries()["country"] == country)
            & (self.get_all_timeseries()["county"].isna())
        ]
        # Second pull all county data for the state
        county_data = (
            self.get_all_timeseries()[
                (self.get_all_timeseries()["state"] == state)
                & (self.get_all_timeseries()["country"] == country)
                & (self.get_all_timeseries()["county"].notna())
            ][["date", "country", "state", "cases", "deaths", "recovered", "active"]]
            .groupby(["date", "country", "state"], as_index=False)[
                ["cases", "deaths", "recovered", "active"]
            ]
            .sum()
        )
        # Now we fill in whatever gaps we can in the state data using the county data
        curr_date = state_data[
            "date"
        ].min()  # Start on the first date of state data we have
        county_data_to_insert = []
        while curr_date > self.start_date:
            curr_date -= datetime.timedelta(days=1)
            # If there is no state data for a day, we need to get some country data for the day
            if len(state_data[state_data["date"] == curr_date]) == 0:
                county_data_for_date = copy(
                    county_data[county_data["date"] == curr_date]
                )
                if (
                    len(county_data_for_date) == 0
                ):  # If there's no county data, we're SOL.
                    continue  # TODO: Revisit. This should be more intelligent
                county_data_for_date = county_data_for_date.iloc[0]
                new_state_row = copy(
                    state_data.iloc[0]
                )  # Copy the first row of the state data to get the right format
                new_state_row["date"] = county_data_for_date["date"]
                new_state_row["cases"] = county_data_for_date["cases"]
                new_state_row["deaths"] = county_data_for_date["deaths"]
                new_state_row["recovered"] = county_data_for_date["recovered"]
                new_state_row["active"] = county_data_for_date["active"]
                county_data_to_insert.append(copy(new_state_row))
        state_county_data = state_data.append(
            pd.DataFrame(county_data_to_insert)
        ).sort_values("date")
        return state_county_data

    def get_timeseries_by_country_state(self, country, state, model_interval):
        #  Prepare a state-level dataset that uses county data to fill in any potential gaps
        return self.prep_data(
            self.combine_state_county_data(country, state), model_interval
        )

    def get_timeseries_by_country(self, country):
        return self.get_all_timeseries()[
            self.get_all_timeseries()["country"] == country
        ]

    def get_population_by_country_state(self, country, state):
        matching_pops = self.get_all_population()[
            (self.get_all_population()["state"] == state)
            & (self.get_all_population()["country"] == country)
        ]
        try:
            return int(matching_pops.iloc[0].at["population"])
        except IndexError as e:
            logging.error("No population data for {}, {}".format(state, country))
            raise e

    def get_beds_by_country_state(self, country, state):
        matching_beds = self.get_all_beds()[
            (self.get_all_beds()["state"] == state)
            & (self.get_all_beds()["country"] == country)
        ]
        beds_per_mille = float(matching_beds.iloc[0].at["bedspermille"])
        return int(
            round(
                beds_per_mille
                * self.get_population_by_country_state(country, state)
                / 1000
            )
        )
