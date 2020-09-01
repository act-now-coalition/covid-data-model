import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta

from pyseir.rt.constants import InferRtConstants


class HistoricalData:
    raw = pd.read_csv("test/data/historical/merged_results_2020_08_19.csv", parse_dates=["date"])
    data = raw[
        [
            "date",
            "state",
            "aggregate_level",
            "cases",
            "deaths",
            "positive_tests",
            "negative_tests",
            "current_hospitalized",
            # "hospital_beds_in_use_any",
            # "contact_tracers_count",
            "Rt_MAP_composite",
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
        start_date = HistoricalData.ref_date + timedelta(days=t_list[0])
        end_date = HistoricalData.ref_date + timedelta(days=t_list[-1])
        rows = HistoricalData.data[
            (HistoricalData.data["state"] == state)
            & (HistoricalData.data["aggregate_level"] == "state")
        ]
        rows["t"] = [(d - HistoricalData.ref_date).days for d in rows["date"]]
        rows = rows.set_index("t")

        rows["rt"] = 1.0 * rows["Rt_MAP_composite"]
        rows["nC"] = rows["positive_tests"] - rows["positive_tests"].shift(fill_value=0.0)
        rows["nC"] = HistoricalData.smooth(rows["nC"])
        cum_tests = rows["positive_tests"] + rows["negative_tests"]
        rows["tests"] = cum_tests - cum_tests.shift(fill_value=0.0)
        rows["tests"] = HistoricalData.smooth(rows["tests"])

        rows["H"] = rows["current_hospitalized"]
        rows["H"] = HistoricalData.smooth(rows["H"])
        rows["nD"] = rows["deaths"] - rows["deaths"].shift(fill_value=0.0)
        rows["nD"] = HistoricalData.smooth(rows["nD"])

        # Apply date filter last
        rows = rows[(rows["date"] >= start_date) & (rows["date"] <= end_date)]

        if compartments_as_functions:
            return (
                lambda t: rows["rt"][t],
                lambda t: rows["nC"][t],
                lambda t: rows["tests"][t],
                lambda t: rows["H"][t],
                lambda t: rows["nD"][t],
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
                lambda t: rows["rt"][t],
                rows["nC"],
                lambda t: rows["tests"][t],
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
    dr = math.log(growth_ratio) * InferRtConstants.SERIAL_PERIOD / (len(t_list) - 1)
    adj_r_f = lambda t: rt_f(t) + dr

    return (average_R, growth_ratio, adj_r_f)
