import numpy as np
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

    # TODO check high values seem to break some states - was VI broken?
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
            # TODO experiment with removing effect of big outbreaks as this is automatically
            # calculated already - avoid "double" counting and applying to states where not happening
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


# These should all belong to the Transitions class but there are problems with creating
# static variables that are calculated from other static variables

# First started with BC data
FH_BY_DECADE_BC = [0.04, 0.01, 0.04, 0.07, 0.11, 0.12, 0.25, 0.5, 0.33, 0.25]
CFR_BY_DECADE_BC = [0.0, 0.0, 0.0, 0.0, 0.004, 0.009, 0.04, 0.125, 0.333, 0.333]

DEATHS_PRCNT_BY_2DECADE_IA = [0.0022, 0.0178, 0.090, 0.411, 0.479]
CASES_PRCNT_BY_2DECADE_IA = [0.10, 0.46, 0.28, 0.12, 0.04]
CFR_BY_2DECADE_IA = [
    1116.0 / 64888.0 * DEATHS_PRCNT_BY_2DECADE_IA[i] / CASES_PRCNT_BY_2DECADE_IA[i]
    for i in range(0, 5)
]

# AB data from https://www.cbc.ca/news/canada/calgary/alberta-covid-19-hospital-icu-average-stay-1.5667884
# Note AB hospitalization rate much lower for young people, higher for middle age
FH_BY_DECADE_AB = [0.005, 0.009, 0.008, 0.021, 0.027, 0.062, 0.112, 0.294, 0.226, 0.226]

# It possible to switch to using data from different jurisdictions. Change the reference to point to different
# per jurisdiction values above to make that happen. BC data appears best (broad range of ages covered) so far
# TODO FUTURE add a test that demonstrates this, or that optimizes these functions for agreement with historical data
FH = FH_BY_DECADE_BC
CFR = CFR_BY_DECADE_BC  # CFR_BY_2DECADE_IA

FH_BY_AGE = [
    (FH[0] + FH[1] + FH[2] + 0.5 * FH[3]) / 3.5,
    (0.5 * FH[3] + FH[4] + FH[5] + 0.5 * FH[6]) / 3,
    (0.5 * FH[6] + FH[7] + FH[8] + FH[9]) / 3.5,
]
CFR_BY_AGE = [  # by 2 decades
    (CFR[0] + 0.75 * CFR[1]) / 1.75,
    (0.25 * CFR[1] + CFR[2] + 0.25 * CFR[3]) / 1.5,
    (0.75 * CFR[3] + CFR[4]) / 1.75,
]
if len(CFR) > 5:  # by decade
    CFR_BY_AGE = [
        (CFR[0] + CFR[1] + CFR[2] + 0.5 * CFR[3]) / 3.5,
        (0.5 * CFR[3] + CFR[4] + CFR[5] + 0.5 * CFR[6]) / 3,
        (0.5 * CFR[6] + CFR[7] + CFR[8] + CFR[9]) / 3.5,
    ]

FD_BY_AGE = [CFR_BY_AGE[i] / FH_BY_AGE[i] for i in range(0, 3)]


class Transitions:
    # Transition fractions to Hospitalizations and Deaths by decade
    # to age groups: 0-35, 35-65 and 65-100

    @staticmethod
    def fh0_f(f_young=None, f_old=None, median_age=None):
        """
        Calculates the fraction of (C)ases and (I)nfections (not tested) that
        will end up being (H)ospitalized.
        """
        rtn = Transitions._interpolate_fractions(FH_BY_AGE, f_young, f_old, median_age)
        return rtn

    @staticmethod
    def fd0_f(f_young=None, f_old=None, median_age=None):
        """
        Calculates the fraction of (C)ases and (I)nfections (not tested) that
        will end up being (H)ospitalized.
        """
        rtn = Transitions._interpolate_fractions(FD_BY_AGE, f_young, f_old, median_age)
        return rtn

    @staticmethod
    def _interpolate_fractions(age_bins, f_young=None, f_old=None, median_age=None):
        """
        Interpolates fractions (fd, fh) over age distributions in 3 bins: young, old, and middle

        Inputs
        - f_young - observed fraction of cases below the age of 35
        - f_old - observed faction of cases above the age of 65
        Either f_young and f_old are specified, or mediang_age.

        Returns fraction in range [0.,1.]
        """
        if f_young is not None and f_old is not None:
            fractions = [f_young, (1.0 - f_young - f_old), f_old]
        elif median_age is not None:
            fractions = Demographics.age_fractions_from_median(median_age)
        else:
            fractions = Demographics.default()

        rtn = sum([fractions[i] * age_bins[i] for i in range(0, 3)])
        return rtn
