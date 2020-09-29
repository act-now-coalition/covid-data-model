import pandas as pd
import math
from datetime import datetime, timedelta

import us
from pyseir.rt.constants import InferRtConstants

# Historical data file, prepared in each API snapshot, including key metrics and inferred Rt values
# TODO update process to save this periodically in this locaiton
# TODO find out why VI data appears to be broken in latest file
HISTORICAL_DATA_FILE = "test/data/historical/merged_results.csv"

# Forecast data
FORECAST_DATA_FILE = "test/data/forecasts/timeseries-common.csv"

# Manually label bad data corrections so they can be automatically applied
DATA_CORRECTIONS = {
    "TX": {"nD": [(208, 180.0)]},
    "NJ": {"nD": [(176, 30.0), (189, 15.0)]},
    "NY": {"nD": [(181, 25.0), (218, 10.0)]},
    "MI": {"nD": [(156, 20.0)]},
}
# TODO FUTURE there are lots more problems with nD (daily) showing as <0.
# - t=218 is big such case for NY

# Most states outbreaks start around calendard day 90. Some are significantly earlier or later
EARLY_OUTBREAK_START_DAY_BY_STATE = {
    "NY": 75,
    "NJ": 80,
    "CT": 80,
    "WY": 100,
    "AK": 115,
    "AZ": 100,
    "DC": 100,
    "CA": 115,
    "GA": 120,
    "IL": 95,
    "IN": 110,
    "KY": 95,
    "MD": 106,
    "ME": 98,
    "MI": 95,
    "MS": 105,
    "MT": 95,
    "NH": 100,
    "NV": 95,
    "OH": 120,
    "PA": 95,
    "SD": 110,
    "UT": 123,
    "VA": 95,
    "WI": 97,
    "WV": 95,
    # TODO address VI that seems to not work and make this globally usable data
}


class HistoricalData:
    """
    Look up historical data to use for testing purposes. Includes both raw data and some
    of our inferred or derived data (e.g. R(t) Bettencourt)
    """

    raw = pd.read_csv(HISTORICAL_DATA_FILE, parse_dates=["date"])
    data = raw[
        [
            "date",
            "state",
            "aggregate_level",  # state, country or metropolitan area
            "cases",
            "deaths",  # totals need to delta for daily
            "positive_tests",  # total, delta is new cases
            "negative_tests",  # total, delta used for positivity
            "current_hospitalized",
            # "hospital_beds_in_use_any",
            # "contact_tracers_count",
            "Rt_MAP_composite",  # Bettencourt R(t)
        ]
    ]
    ref_date = datetime(2020, 1, 1)

    @staticmethod
    def smooth(series):
        smoothed = series.rolling(
            InferRtConstants.COUNT_SMOOTHING_WINDOW_SIZE,
            win_type="gaussian",
            min_periods=InferRtConstants.COUNT_SMOOTHING_KERNEL_STD,
            center=True,
        ).mean(std=InferRtConstants.COUNT_SMOOTHING_KERNEL_STD)
        return smoothed

    @staticmethod
    def get_state_data_for_dates(
        state, t_list, compartments_as_functions=False, no_functions=False
    ):
        """
        Get summary state data from prepared file
        """

        start_date = HistoricalData.ref_date + timedelta(days=t_list[0])
        end_date = HistoricalData.ref_date + timedelta(days=t_list[-1])

        # also use early start date (30 days earlier) when collecting data for functions that might
        # be evaluated delayed (at earlier times)
        early_start = start_date + timedelta(days=-30)

        rows = HistoricalData.data[
            (HistoricalData.data["state"] == state)
            & (HistoricalData.data["aggregate_level"] == "state")
        ].copy()
        rows["t"] = [(d - HistoricalData.ref_date).days for d in rows["date"]]
        rows = rows.set_index("t")

        rows["rt"] = 1.0 * rows["Rt_MAP_composite"]
        rows["nC"] = rows["positive_tests"] - rows["positive_tests"].shift(fill_value=0.0)

        cum_tests = rows["positive_tests"] + rows["negative_tests"]
        rows["tests"] = cum_tests - cum_tests.shift(fill_value=0.0)

        rows["H"] = rows["current_hospitalized"]
        rows["nD"] = rows["deaths"] - rows["deaths"].shift(fill_value=0.0)

        # Apply fixes before smoothing
        if state in DATA_CORRECTIONS:
            state_corrections = DATA_CORRECTIONS[state]
            for col in state_corrections:
                for (day, value) in state_corrections[col]:
                    # rows[col][day] = value - but generates SettingWithCopyWarning
                    rows.loc[day, col] = value

        # Then smooth the data
        rows["nC"] = HistoricalData.smooth(rows["nC"])
        rows["tests"] = HistoricalData.smooth(rows["tests"])
        rows["H"] = HistoricalData.smooth(rows["H"])
        rows["nD"] = HistoricalData.smooth(rows["nD"])

        # Apply date filter last
        early_rows = rows[(rows["date"] >= early_start) & (rows["date"] <= end_date)]
        rows = early_rows[(early_rows["date"] >= start_date)]

        if compartments_as_functions:
            return (
                lambda t: early_rows["rt"][t],
                lambda t: early_rows["nC"][t],
                lambda t: early_rows["tests"][t],
                lambda t: early_rows["H"][t],
                lambda t: early_rows["nD"][t],
            )
        elif no_functions:
            return (
                rows["rt"],
                rows["nC"],
                rows["tests"],
                rows["H"],
                rows["nD"],
            )
        else:
            return (
                lambda t: early_rows["rt"][t],
                rows["nC"],
                lambda t: early_rows["tests"][t],
                rows["H"],
                rows["nD"],
            )

    @staticmethod
    def get_states():
        return HistoricalData.data["state"].unique()


def adjust_rt_to_match_cases(rt_f, new_cases, t_list):
    """
    Given R(t) and new_cases(t) series and a time range, find adjustment to R(t)
    so that it properly predicts the ratio of ending/starting new_cases
    """
    rt = [rt_f(t) for t in t_list]
    average_R = sum(rt) / len(rt)
    case_growth = new_cases(t_list[-1]) / new_cases(t_list[0])
    r_1_growth = math.exp((average_R - 1) * (len(t_list) - 1) / InferRtConstants.SERIAL_PERIOD)
    growth_ratio = case_growth / r_1_growth
    if growth_ratio > 0:
        dr = math.log(growth_ratio) * InferRtConstants.SERIAL_PERIOD / (len(t_list) - 1)
    else:  # something funny so skip
        dr = 0.0
    adj_r_f = lambda t: rt_f(t) + dr

    return (average_R, growth_ratio, adj_r_f)


class ForecastData:
    """
    Look up forecast data to use for testing purposes.
    """

    raw = pd.read_csv(FORECAST_DATA_FILE, parse_dates=["date"])
    data = raw[
        [
            "fips",
            "date",
            "weekly_new_cases_0.5",  # 50% confidence from many inputs
            "weekly_new_deaths_0.5",  # 50% confidence from many inputs,
        ]
    ]

    data = data[data["fips"] < 100]
    ref_date = datetime(2020, 1, 1)

    @staticmethod
    def get_state_nC_nD_forecast(state):
        state_obj = us.states.lookup(state)
        fips = int(state_obj.fips)

        # Get the right rows
        rows = ForecastData.data[(ForecastData.data["fips"] == fips)]
        rows = rows.drop(columns="fips")
        rows["date"] = pd.to_datetime(rows["date"])
        rows = rows.set_index("date")
        rows.columns = ["nC", "nD"]

        # Format the data as tuples
        nC_forecast = [
            ((index - ForecastData.ref_date).days, value / 7.0)
            for (index, value) in rows.nC.items()
        ]
        nD_forecast = [
            ((index - ForecastData.ref_date).days, value / 7.0)
            for (index, value) in rows.nD.items()
        ]

        return (nC_forecast, nD_forecast)
