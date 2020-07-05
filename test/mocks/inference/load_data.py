import numpy as np
from math import exp
from collections import namedtuple
from enum import Enum

import pandas as pd

from pyseir.load_data import HospitalizationDataType
from pyseir.rt.constants import InferRtConstants

"""
This module stubs out pyseir.load_data for testing purposes. It returns special data examples
for specific tests.
"""


class DataGeneratorType(Enum):
    EXP = "exponential"
    LIN = "linear"


RateChange = namedtuple("RateChange", "t0 reff")
DataSpec = namedtuple("DataSpec", "generator_type disable_deaths scale ratechange1 ratechange2")


class DataGenerator:
    """
    Generates data according to a sequence of growth rates that kick in at
    various times (assumed to be integers starting from 0 and supplied in order).
    Growth rate of 0. implies a constant value
    """

    def __init__(self, spec):
        self.generator_type = spec.generator_type
        self.disable_deaths = spec.disable_deaths
        self.scale = spec.scale
        self.growth_rate = None
        self.t0 = None
        self.last_value = spec.scale

        self.rate_at = {}
        for change in [spec.ratechange1, spec.ratechange2]:
            if change is None:
                continue
            if self.generator_type == DataGeneratorType.EXP:
                self.rate_at[change.t0] = (change.reff - 1.0) / InferRtConstants.SERIAL_PERIOD
            else:
                self.rate_at[change.t0] = change.reff

    def generate_data(self, time):
        if time in self.rate_at:
            self.t0 = time
            self.growth_rate = self.rate_at[time]
            self.scale = self.last_value

        if self.generator_type == DataGeneratorType.EXP:  # exponential growth
            self.last_value = 1.0 * self.scale * exp(self.growth_rate * (time - self.t0))
        else:  # linear growth
            self.last_value = self.scale + self.growth_rate * (time - self.t0)
        return self.last_value


def _get_cases_for_times(generator, times):
    return np.array(list(map(generator.generate_data, times)))


def create_synthetic_df(data_generator):
    """
    Generates case and death data.
    """
    times = list(range(0, 100))
    dates = pd.date_range("2020-01-01", periods=100)
    observed_new_cases = _get_cases_for_times(data_generator, times)

    if data_generator.disable_deaths:
        observed_new_deaths = np.zeros(len(times))
    else:
        observed_new_deaths = 0.03 * observed_new_cases

    df = pd.DataFrame(data=dict(cases=observed_new_cases, deaths=observed_new_deaths), index=dates)
    return df


# def load_hospitalization_data_by_state(state, t0=None):
#     data_generator = DataGenerator(specs[state])
#     times = list(range(0, 100))
#     observed_new_cases = _get_cases_for_times(data_generator, times)
#
#     if data_generator.disable_deaths:
#         hospitalizations = np.zeros(len(times))
#     else:
#         hospitalizations = 0.12 * observed_new_cases
#     return (times, hospitalizations, HospitalizationDataType.CURRENT_HOSPITALIZATIONS)


# _________________Other methods to mock__________________
# (
#                 self.times,
#                 self.observed_new_cases,
#                 self.observed_new_deaths,
#             ) = self.load_data.load_new_case_data_by_fips(
#                 self.fips,
#                 t0=self.ref_date,
#                 include_testing_correction=self.include_testing_correction,
#             )

# (
#                 self.hospital_times,
#                 self.hospitalizations,
#                 self.hospitalization_data_type,
#             ) = load_hospitalization_data(self.fips, t0=self.ref_date)
