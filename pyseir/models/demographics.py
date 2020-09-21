import numpy as np
import pandas as pd
import math
from enum import Enum

# Not using currently
class ContactsType(Enum):
    DEFAULT = 1
    LOCKDOWN = 2
    RESTRICTED = 3


# Helper function
def extend_tlist_at_beginning(tlist, days):
    """
    Extend a time list at the beginning
    """
    start = int(tlist[0]) - days
    end = int(tlist[-1])
    return np.linspace(start, end, end - start + 1)


class Demographics:
    """
    Helper class for demographics information and how to map it onto age groups
    used in adjusting hospitalization and deaths fractions.
    """

    # TODO check high values seem to break some states
    MEDIAN_RATE_CONSTANT = 1.5  # slightly better than 1.25 and 1.4
    DEFAULT_DRIVER_AGE = 25.0

    # Note these contact matrices are only used in testing so far
    HOUSEHOLD = np.array([[0.2, 0.2, 0.0], [0.2, 0.2, 0.0], [0.0, 0.0, 0.0]])
    SCHOOLS = np.array([[2.0, 0.2, 0.0], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0]])
    ESSENTIAL = np.array([[0.1, 0.1, 0.05], [0.1, 0.1, 0.05], [0.05, 0.05, 0.0]])
    LT_VISITS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.5]])
    SOCIAL_1 = np.array([[1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.1]])
    SOCIAL_2 = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])

    CONTACTS = {
        ContactsType.DEFAULT: (HOUSEHOLD + SCHOOLS + ESSENTIAL + LT_VISITS + SOCIAL_2),
        ContactsType.LOCKDOWN: (ESSENTIAL + HOUSEHOLD),
        ContactsType.RESTRICTED: (HOUSEHOLD + SCHOOLS + ESSENTIAL + SOCIAL_1),
    }

    def __init__(
        self, young=None, old=None, median_age=None, count=1000.0, dwell_time=None, model=None
    ):

        from pyseir.models.nowcast_seir_model import NowcastingSEIRModel

        if median_age is not None:  # can be initialized with just a median age
            self.median_age = median_age
        elif young is not None and old is not None:  # or with young and old
            assert young >= 0.0 and old >= 0.0 and young + old <= 1.0
            self.young = young
            self.old = old
            self.medium = 1.0 - young - old
        else:
            assert False

        if model is None:
            model = NowcastingSEIRModel()
        if dwell_time is None:
            self.dwell_time = model.serial_period
        else:
            self.dwell_time = dwell_time

        self.count = count
        self.driver_age = Demographics.DEFAULT_DRIVER_AGE

    def get_median_age(self):
        rtn = None
        if self.median_age is not None:
            rtn = self.median_age
        else:
            if self.young > 0.5:
                rtn = 35.0 * (self.young - 0.5)
            elif self.old > 0.5:
                rtn = 65.0 + 35.0 * (self.old - 0.5)
            else:
                rtn = 30.0 / self.medium * (0.5 - self.young) + 35.0
        return min(65.0, max(35.0, rtn))

    def as_array(self):
        """
        Return numpy array with the three classes as relative fractions
        """
        return np.array([self.young, self.medium, self.old])

    def from_array(self, arr):
        """
        Update three classes from numpy array
        """
        (self.young, self.medium, self.old) = list(arr)
        self.median_age = None

    def evolve_median(self, rt, default_age):
        """
        Evolve median age based on assumption that growing cases drives median towards driver_age
        while it naturally decays back to default_age (which can vary with each call) if no growth.
        """
        assert self.median_age is not None and self.driver_age is not None

        # strength of case growth (R(t)>1) drives median agae towards driver_age
        rate = Demographics.MEDIAN_RATE_CONSTANT * (rt - 1.0) / self.dwell_time if rt > 1.0 else 0.0

        new_median = (self.median_age + rate * self.driver_age + default_age / self.dwell_time) / (
            1.0 + rate + 1.0 / self.dwell_time
        )

        self.median_age = new_median

    def update_by_contacting_another(self, contacts_type, adding, from_demo, dt=1.0):
        """
        Update demographics based on new set of individuals being added from a source based on contacts
        """
        raw_new = Demographics.CONTACTS[contacts_type] @ self.as_array() * from_demo.as_array()
        scaled_new = raw_new * (adding / raw_new.sum())
        new_counts = self.count * self.as_array() * (1.0 - dt / self.dwell_time) + scaled_new

        new_fractions = new_counts / new_counts.sum()
        self.from_array(new_fractions)

    @staticmethod
    def infer_median_age_function(t_list, rt_f):
        t_list = extend_tlist_at_beginning(t_list, 30)
        default_usa_f = Demographics.median_age_f()
        demo = Demographics(median_age=default_usa_f(t_list[0]))

        values = {t_list[0]: demo.get_median_age()}
        for t in t_list:
            if t < 135.0:  # don't use this early in the pandemic
                demo = Demographics(median_age=default_usa_f(t))
            else:
                demo.evolve_median(rt_f(t), default_usa_f(t))
            values[t] = demo.get_median_age()

        def ftn(t):
            if t <= t_list[0]:
                return values[t_list[0]]
            elif t >= t_list[-1]:
                return values[t_list[-1]]
            else:
                return values[t]

        return ftn

    @staticmethod
    def default():
        r_lo_mid = 1.1
        r_hi_mid = 0.4
        mid = 1.0 / (1.0 + r_lo_mid + r_hi_mid)
        return [r_lo_mid * mid, mid, r_hi_mid * mid]

    @staticmethod
    def age_fractions_from_median(median_age):
        center_pref = 4.0
        if median_age < 35.0 or median_age > 65.0:
            assert False
        k = (median_age - 35.0) / 30.0

        # Coefficients of quadratic equation for m - fraction of medium age
        a = 2.0 / center_pref + 2 * k * (1.0 - k)
        b = 1.0
        c = -0.5

        # Solution of quadratic equation
        m = (-b + math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)

        # Then solve for young and old
        y = 0.5 - k * m
        o = 1.0 - m - y

        avg_vs_median = (17.5 * y + 50.0 * m + 82.5 * o) - median_age

        return (y, m, o)

    @staticmethod
    def median_age_f(state=None):
        """
        Median age of covid cases for all of US and specific individual states
        """
        if state not in ["FL"]:  # median age for country as a whole
            # From https://www.cdc.gov/coronavirus/2019-ncov/covid-data/covidview/index.html
            data = {
                68: 56.3,
                75: 53.5,
                82: 54.7,
                89: 57.1,
                96: 59.2,
                103: 61.9,
                110: 59.5,
                117: 55.1,
                124: 53.7,
                131: 51.8,
                138: 50.4,
                145: 48.6,
                152: 47.8,
                159: 44.7,
                166: 42.7,
                173: 41.5,
                180: 40.2,
                187: 40,
                194: 40.7,
                201: 39.8,
                208: 40,
                215: 40.5,
                222: 40.7,
                229: 40.6,
            }
        else:
            # From reports in http://ww11.doh.state.fl.us/comm/_partners/covid19_report_archive/
            data = {
                133: 52,
                140: 49,
                147: 43,
                154: 41,
                161: 39,
                168: 36,
                175: 35,
                182: 36,
                189: 39,
                196: 40,
                203: 41,
                210: 41.5,
                217: 42,
                224: 43,
                231: 43,
            }

        days = list(data.keys())
        values = list(data.values())

        def med_func(t):
            if t <= days[0]:
                return values[0]
            elif t >= days[-1]:
                return values[-1]
            else:
                for (day, value) in data.items():
                    if t >= day:
                        last = value
                    else:  # interpolate between values
                        return 1.0 / 7.0 * ((day - t) * last + (t - day + 7) * value)

        return med_func
