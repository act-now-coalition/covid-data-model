import numpy as np
from math import exp
from collections import namedtuple
from enum import Enum

from pyseir.load_data import HospitalizationDataType
from test.mocks.inference.pyseir_default_parameters import pyseir_default_parameters

"""
This module stubs out pyseir.load_data for testing purposes. It returns special data examples
for specific tests. Different state names will be used to control this.
"""

SERIAL_PERIOD = (
    1 / pyseir_default_parameters["sigma"] + 0.5 * 1 / pyseir_default_parameters["delta"]
)


class DataGeneratorType(Enum):
    EXP = "exponential"
    LIN = "linear"


RateChange = namedtuple("RateChange", "t0 reff")
DataSpec = namedtuple("DataSpec", "generator_type disable_deaths scale ratechange1 ratechange2")

# Need this because API for cases and hospitalizations use different state identifiers
# Each of these state generators is defined in a specific unit test
state_to_code = {"New York": "NY", "Hawaii": "HI", "Alaska": "AK", "Alabama": "AL"}

specs = {}
# Initialize a data generator and associate it with a state so can be picked up with regular processing
def initializeStateDataGenerator(state, dataspec):
    specs[state] = dataspec


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
                self.rate_at[change.t0] = (change.reff - 1.0) / SERIAL_PERIOD
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


def load_new_case_data_by_state(state, ref_date, include_testing_correction):
    """
    Generates case and death data.
    Note stupidly called with full name of state
    """
    data_generator = DataGenerator(specs[state_to_code[state]])
    times = list(range(0, 100))
    observed_new_cases = _get_cases_for_times(data_generator, times)

    if data_generator.disable_deaths:
        observed_new_deaths = np.zeros(len(times))
    else:
        observed_new_deaths = 0.03 * observed_new_cases
    return (times, observed_new_cases, observed_new_deaths)


def load_hospitalization_data_by_state(state, t0=None):
    data_generator = DataGenerator(specs[state])
    times = list(range(0, 100))
    observed_new_cases = _get_cases_for_times(data_generator, times)

    if data_generator.disable_deaths:
        hospitalizations = np.zeros(len(times))
    else:
        hospitalizations = 0.12 * observed_new_cases
    return (times, hospitalizations, HospitalizationDataType.CURRENT_HOSPITALIZATIONS)


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
