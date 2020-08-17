import numpy as np
import pandas as pd
import math

# TODO setup JAX instead of numpy
# initial trials showed no real improvement, but I remain optimstic
# from jax.config import config
# config.update("jax_enable_x64", True)
# from jax import numpy as np
# from jax import jit
from scipy.integrate import odeint

import matplotlib.pyplot as plt

z0 = np.array([0])


def derivative(t):
    return np.append(z0, (t[1:] - t[:-1]))


def steady_state_ratios(r, suppression=1.0):

    tlist = np.linspace(0, 60, 61)
    if r > 1:
        initial_infected = 100
    else:
        initial_infected = 10000
    const_suppression = lambda t: suppression

    model = SEIRModel(
        N=10000000,
        t_list=tlist,
        suppression_policy=const_suppression,
        R0=r,
        A_initial=0.7 * initial_infected,
        I_initial=initial_infected,
    )

    model.run()
    last = lambda s: model.results[s][-1]
    lastI = last("I")

    return (
        last("E") / lastI,
        1.0,
        last("A") / lastI,
        last("HGen") / lastI,
        last("HICU") / lastI,
        last("HVent") / lastI,
        last("direct_deaths_per_day") / lastI,
    )


class Demographics:
    """
    Helper class for demographics information and how to map it onto age groups
    used in adjusting hospitalization and deaths fractions.
    """

    r_lo_mid = 1.1
    r_hi_mid = 0.4

    @staticmethod
    def default():
        mid = 1.0 / (1.0 + Demographics.r_lo_mid + Demographics.r_hi_mid)
        return [Demographics.r_lo_mid * mid, mid, Demographics.r_hi_mid * mid]

    @staticmethod
    def age_fractions_from_median(median_age):
        center_pref = 4.0
        assert median_age > 35 and median_age < 65
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
    def median_age_f():
        """
        Median age of covid cases for all of US.
        TODO this is from Florida so need to update.
        """

        def med_func(t):
            t1 = 120.0
            max = 48.0
            t2 = 180.0
            min = 37.0
            if t < t1:
                return max
            elif t > t2:
                return min
            else:
                return max - (max - min) * (t - t1) / (t2 - t1)

        return med_func


def ramp_function(t_list, start_value, end_value):
    """
    Returns a function that implements a linear ramp from start_value to end_value over
    the time domain present in (an ordered) t_list
    """
    rate = (end_value - start_value) / (t_list[-1] - t_list[0])
    ftn = lambda t: start_value + (t - t_list[0]) * rate
    return ftn


# Transition fractions to Hospitalizations and Deaths by age groups
# 0-35, 35-65 and 65-100
# TODO need to normalize contributions per decade by local demographics
# TODO put this code inside NowcastingSEIRModel
FH_BY_AGE = [  # Roughly rom BC data - need to cross check
    (0.04 + 0.01 + 0.04 + 0.5 * 0.07) / 3.5,
    (0.5 * 0.07 + 0.11 + 0.12 + 0.5 * 0.25) / 3,
    (0.5 * 0.25 + 0.5 + 0.33 + 0.25) / 3.5,
]
CFR_BY_AGE = [  # Roughly rom BC data - need to cross check
    0.0,
    (0.5 * 0.0 + 0.004 + 0.009 + 0.5 * 0.04) / 3,
    (0.5 * 0.04 + 0.125 + 0.333 + 0.333) / 3.5,
]
FD_BY_AGE = [CFR_BY_AGE[i] / FH_BY_AGE[i] for i in range(0, 3)]


class NowcastingSEIRModel:
    """
    Simplified SEIR Model sheds complexity where not needed to be accurate enough. See ModelRun and 
    ModelRun._time_step for details
    
    The concept of running the Model has been split off into another class (ModelRun).

    TODO Next steps:
    1) Look at time dependent behaviour, hook up to standard graphing
    2) Turn on delay and see if that helps match peak behaviour
    3) Validate test positivity contribution to peaks against that observed

    See test/seir_model_training_test.py for tests
    """

    def __init__(
        self,
        # ____________These are (to be) trained factors that are passed in___________
        median_age=None,
        lr_fh=(1.0, 0.0),
        lr_fd=(1.0, 0.0),
        delay_ci_h=0,  # added days of delay between infection and hospitalization
    ):
        # Retain passed in parameters
        self.lr_fh = lr_fh
        self.lr_fd = lr_fd
        self.delay_ci_h = delay_ci_h  # not using this yet

        # __________Fixed parameters not trained_________
        self.t_e = 2.0  # 2.0  sigma was 1/3. days^-1 below
        # TODO serial period as ~5 days in our old model runs. How is this not 6 = 3 + 1/2 * 6?
        self.t_i = 6.0  # delta was 1/6. days^-1 below
        self.serial_period = self.t_e + 0.5 * self.t_i  # this is 5 days agrees with delta below
        self.t_h = 8.0  # delta_hospital below was 1./8. days^-1
        self.fw0 = 0.5
        self.pos0 = 0.5  # max positivity for testing -> 0
        self.pos_c = 1.75
        self.pos_x0 = 2.0
        # Solution that ensures continuity of function below and its derivative
        self.pos_b = (3.0 * self.pos_x0 - self.pos_c) / (4.0 * self.pos_x0 ** 1.5)
        self.pos_d = self.pos_x0 ** 0.5 / 4.0 * (3.0 * self.pos_c - self.pos_x0)

    def run_stationary(self, rt, median_age, t_over_x, x_is_new_cases=True):
        """
        Given R(t) and T/I or T/C run to steady state and return ratios of all compartments
        """

        x_fixed = 1000.0
        num_days = 100
        t_list = np.linspace(0, num_days, num_days + 1)

        if x_is_new_cases:
            run = ModelRun(
                self,
                N=2e7,
                t_list=t_list,
                testing_rate_f=lambda t: t_over_x * x_fixed,
                rt_f=lambda t: rt,
                case_median_age_f=lambda t: median_age,
                nC_initial=x_fixed,
                force_stationary=True,
            )
        else:
            i_fixed = 1000.0
            run = ModelRun(
                self,
                N=2e7,
                t_list=make_tlist(100),
                testing_rate_f=lambda t: t_over_x * x_fixed,
                rt_f=lambda t: rt,
                case_median_age_f=lambda t: median_age,
                I_initial=x_fixed,
                force_stationary=True,
            )

        (history, ratios) = run.execute_lists_ratios()
        compartments = history[-1]
        ratios["rt"] = rt

        return (ratios, compartments)

    def positivity(self, t_over_i):
        """
        Test positivity as a function of T = testing rate / I = number of infected
        This function should match two constraints
            p(x) = .5 for x->0
            p(x) <~ 1/x for x-> infinity (almost all infections found as testing -> infinity)
        To achieve this different functions are used (switching at x0) and the constants b and d
        are solved for to ensure continuity of the function and its derivative across x0 
        """
        p = 0.5 / t_over_i
        if t_over_i < self.pos_x0:  # approaches .5 for x -> 0
            p = p * (t_over_i - self.pos_b * t_over_i ** 1.5)
        else:  # approaches .875/x for x -> infinity
            p = p * (self.pos_c - self.pos_d / t_over_i ** 0.5)
        return p

    def positivity_to_t_over_i(self, pos):
        """
        Rely on positivity to be a continuous, strictly decreasing function of t_over_i over [.001,1000]
        to invert using binary search in log space
        """
        lo = -3.0
        if pos > self.positivity(10.0 ** lo):
            return 10.0 ** lo
        hi = +3.0
        if pos < self.positivity(10.0 ** hi):
            return 10.0 ** hi
        eps = 0.01
        while hi - lo > eps:
            x = 0.5 * (hi + lo)
            if pos < self.positivity(10.0 ** x):
                lo = x
            else:
                hi = x
        return 10.0 ** x

    def adjustFractions(self, adj_H, adj_nD):
        """
        Adjusting the fractions (as a model is running)
        """
        (b, m) = self.lr_fh
        self.lr_fh = (b * adj_H, m)
        (b, m) = self.lr_fd
        self.lr_fd = (b * adj_nD, m)

    def fh0_f(self, f_young=None, f_old=None, median_age=None):
        """
        Calculates the fraction of (C)ases and (I)nfections (not tested) that
        will end up being (H)ospitalized.
        """
        rtn = self._interpolate_fractions(FH_BY_AGE, f_young, f_old, median_age)
        return rtn

    def fd0_f(self, f_young=None, f_old=None, median_age=None):
        """
        Calculates the fraction of (C)ases and (I)nfections (not tested) that
        will end up being (H)ospitalized.
        """
        rtn = self._interpolate_fractions(FD_BY_AGE, f_young, f_old, median_age)
        return rtn

    def _interpolate_fractions(self, age_bins, f_young=None, f_old=None, median_age=None):
        """
        Interpolates fractions (fd, fh) over age distributions in 3 bins: young, old, and middle

        Inputs
        - f_young - observed fraction of cases below the age of 35
        - f_old - observed faction of cases above the age of 65
        Either f_young and f_old are specified, or mediang_age.

        Returns fraction in range [0.,1.]
        """
        if f_young is not None and f_old is not None:
            fractions = [f_young, (1.0 - f_yound - f_old), f_old]
        elif median_age is not None:
            fractions = Demographics.age_fractions_from_median(median_age)
        else:
            fractions = Demographics.default()

        rtn = sum([fractions[i] * age_bins[i] for i in range(0, 3)])
        return rtn


class ModelRun:
    """
    A run (in time) of a model with (previously fit) parameters maintained in model
    Run will also have the ability to "linearly" adjust the observed ratios of hopitalizations to cases and deaths
    to hopitalizations to align with observed compartment histories without external shimming.

    Model compartments (evolved in time):
        S - susceptible (its really S/N that is important)
        E - exposed -> will become I or A over time
        I - infected, syptomatic -> can become C over time
        W - infected, weakly or asympomatic
        nC - new (confirmed) cases for tracking purposes
        C - observed (confirmed) cases -> can become H over time
        H - observed hosptializations (usually observable). 
        D - observed deaths
        R - recoveries (just to validate conservation of people)
    Note R and HICU, HVent (generated from H - but outside of this model) are not included.

    Observables, recent history smoothed and/or future projections, EVENTUALLY to be used to either inject known
    important sources of time variation into, or constrain, the running model:
        rt - growth rate for C (TODO will we adjust this to apply to I+C?)
        case_median_age - allows for coarse adjustments for differing responses of age groups.
            Might generalize to case_fraction_young, case_fraction_old with case_fraction_young + case_fraction_old < 1
        test_positivity - fraction of tests that are positive
        C, H, D described above. Note its the ratios H/C and D/H that provide constraints
        test_processing_delay - in days from time sample supplied to test result available
        case_fraction_traceable - fraction of cases that have been traced to their source
    
    Linear (ramp in time) adjustments (to I/C -> H and H->D rates) determined during model training.

    Parameters
    ----------
    model: NowcastingSEIRModel
        Run independent parts of the base Model - parts that will be trained generally ahead of time
    N: int
        Total population TODO move this to the Model
    t_list: int[]
        Times relative to a reference date
    rt_f: lambda(t)
        instead of suppression_policy, R0
        TODO option to provide smoothed cases directly
    testing_rate_f: lambda(t)
        Testing rate
    case_median_age_f: lambda(t)
        Median age of people that test positive each day

    TODO how to specify compartments for initialization (up to today) that covers all cases
        Starting from bare minimum of values: C or I
        Starting from just a few observables for today: C, D, nC, H, nD (last 3 as constraints)
        Starting from DataFrame outputs of a previous run

    today: int[]
        Day that represents "today". Days before this day are treated as burn in for the run while those
        after it are the "predictions" of the future. When this parameter is not supplied the whole
        run is treated as in the future (today is the 1st day in t_list)
    force_stationary: boolean
        This run used to determine steady state compartment distributions to initialize another run.
        If True compartments are adjusted on each timestep to keep S and one of (I, nC) constant
    auto_initialize_other_compartments: boolean
        Controls (primarily non observable) compartment initialization. If set this flag causes a run
        with force_stationary to be run to derive steady state compartment values for initialization
    """

    @staticmethod
    def array_to_df(arr):
        df = pd.DataFrame(arr, columns=["S", "E", "I", "A", "nC", "C", "H", "nD", "R", "b"])
        if len(df.index) > 1:
            df["D"] = df["nD"].cumsum()
            df["b"][0] = None
        return df

    @staticmethod
    def df_to_array(df):
        rtn = df[["S", "E", "I", "A", "nC", "C", "H", "nD", "R", "b"]].to_numpy()
        return rtn

    @staticmethod
    def dict_to_array(d):
        df = pd.DataFrame(d, index=[0])
        return ModelRun.df_to_array(df)[0]

    @staticmethod
    def array_to_dict(arr):
        rtn = {
            "S": arr[0],
            "E": arr[1],
            "I": arr[2],
            "A": arr[3],
            "nC": arr[4],
            "C": arr[5],
            "H": arr[6],
            "nD": arr[7],
            "R": arr[8],
            "b": arr[9],
        }
        return rtn

    @staticmethod
    def make_array(S, I=None, nC=None, H=None, nD=None):
        if nC is not None:
            nC0 = nC
            I0 = nC
        elif I is not None:
            I0 = I
            nC0 = 0.0
        else:
            I0 = 100.0
            nC0 = 0.0
        y = [
            (
                S,
                0.0,  # E
                I0,
                0.0,  # A=W
                nC0,
                7.0 * nC0,  # C
                0.0 if H is None else H,  # H
                0.0 if nD is None else nD,  # D
                0.0,  # R
                0.0,  # beta
            )
        ]
        return y

    def __init__(
        self,
        model,
        N,  # population of the jurisdiction
        t_list,  # array of days
        #### Time varying inputs needed into the future
        testing_rate_f,  # tests per day assumed for the future
        rt_f,  # instead of suppression_policy, R0
        # TODO option to provide smoothed cases directly
        case_median_age_f=None,
        # case_fraction_traceable_f=None,
        # test_processing_delay_f=None,
        test_positivity_f=None,
        #### Initial conditions of major observables
        I_initial=None,  # initial infected
        nC_initial=None,
        S_initial=None,
        nD_initial=None,
        H_initial=None,
        initial_compartments=None,
        compartment_ratios_initial=None,
        # hospitalizations_threshold=None,  # At which point mortality starts to increase due to constrained resources
        # Observable compartments to be eventually used in further constraining the model
        # observed_compartment_history=None,
        #### Optional controls for how the model run operates
        force_stationary=False,  # if True susceptible will be pinned to N
        auto_initialize_other_compartments=False,
        auto_calibrate=False,
        historical_compartments=None,
        today=None,
    ):
        self.model = model
        self.N = N
        self.t_list = t_list
        self.testing_rate_f = testing_rate_f
        self.rt_f = rt_f
        self.force_stationary = force_stationary
        self.test_positivity_f = test_positivity_f
        self.case_median_age_f = case_median_age_f
        self.auto_calibrate = auto_calibrate
        self.initial_compartments = initial_compartments
        self.historical_compartments = historical_compartments
        self.today = today

        t0 = t_list[0]
        if self.historical_compartments is not None:
            hc = self.historical_compartments
            S_initial = hc["S"].values[0] if "S" in hc else None
            I_initial = hc["I"].values[0] if "I" in hc else None
            nC_initial = hc["nC"].values[0] if "nC" in hc else None
            H_initial = hc["H"].values[0] if "H" in hc else None
            nD_initial = hc["nD"].values[0] if "nD" in hc else None
        elif self.initial_compartments is not None:
            ic = self.initial_compartments
            S_initial = ic["S"] if "S" in ic else None
            I_initial = ic["I"] if "I" in ic else None
            nC_initial = ic["nC"] if "nC" in ic else None
            H_initial = ic["H"] if "H" in ic else None
            nD_initial = ic["nD"] if "nD" in ic else None

        if auto_initialize_other_compartments:
            x = nC_initial if nC_initial is not None else I_initial
            (ignore, compartments) = model.run_stationary(
                rt_f(t0),
                case_median_age_f(t0),
                t_over_x=testing_rate_f(t0) / x,
                x_is_new_cases=True if nC_initial is not None else False,
            )
            self.compartment_ratios_initial = compartments
        else:
            self.compartment_ratios_initial = compartment_ratios_initial  # get applied at run time

        S = S_initial if S_initial is not None else self.N
        self.history = ModelRun.make_array(
            S=S, I=I_initial, nC=nC_initial, H=H_initial, nD=nD_initial
        )

    def execute_dataframe_ratios_fig(self):
        (history, ratios) = self.execute_lists_ratios()
        df = ModelRun.array_to_df(history)
        self.results = df
        fig = self.plot_results()
        return (df, ratios, fig)

    def execute_lists_ratios(self):
        y = self.history[0]
        y0 = ModelRun.array_to_dict(y)  # convenient to use dict
        change_track_nC = True if y0["nC"] > 0.0 else False

        if self.compartment_ratios_initial is not None:
            y0 = ModelRun.array_to_dict(self.history[0])
            # TODO do the same for compartments
            if y0["nC"] > 0.0:  # starting from all compartments scaled
                factor = y0["nC"] / self.compartment_ratios_initial[4]  # cases
            else:
                factor = y0["I"] / self.compartment_ratios_initial[2]  # cases
            y = [x * factor for x in self.compartment_ratios_initial]
            y[0] = y0["S"]
            y[8] = 0.0  # start recoveries over
            y[9] = 0.0

        if self.auto_calibrate:
            current = ModelRun.array_to_dict(y)
            adj_H = 1.0
            adj_nD = 1.0
            if y0["H"] > 0.0 and current["H"] > 0.0:
                adj_H = y0["H"] / current["H"]
                current["H"] = y0["H"]
            if y0["nD"] > 0.0 and current["nD"] > 0.0:
                adj_nD = y0["nD"] / current["nD"] / adj_H
                current["nD"] = y0["nD"]
            y = ModelRun.dict_to_array(current)
            self.model.adjustFractions(adj_H, adj_nD)

        y_accum = list()
        y_accum.append(y)

        implicit = False if self.force_stationary else True

        # Iterate over all time steps
        for t in self.t_list[:-1]:
            # TODO determine dt from t_list and make sure _time_step using it correctly
            y = list(y)
            dy = self._time_step(y, t, dt=1.0, implicit_infections=implicit)
            (dS, dE, dI, dW, nC, dC, dH, dD, dR, b) = dy
            y_new = [a + b for a, b in zip(y, dy)]
            # do not accumulate daily new cases, deaths or beta
            y_new[4] = nC
            y_new[7] = dD
            y_new[9] = b

            if change_track_nC:
                fractional_change = y_new[4] / y[4]
            else:
                fractional_change = y_new[2] / y[2]

            # Apply stationarity if applicable
            if self.force_stationary:
                S_last = y[0]
                # apply fractional change adjustments except to S and b(eta)
                y = [v / fractional_change for v in y_new]
                y[9] = y_new[9]
                y[0] = S_last

                # Ignore accumulating
                # D = dD
                # y[7] = D
                R = dR
                y[8] = R
            else:
                y = y_new

            (S, E, I, W, nC, C, H, nD, R, b) = y
            r_dD_nC = nD / nC  # new deaths over new cases - apparent time dependent IFR
            r_C_WIC = C / (W + I + C)  # fraction of sick that are official cases
            r_C_IC = C / (I + C)  # fraction of true cases that are officially counted
            r_H_IC = H / (C + I)  # hospitalizations per true case
            r_dD_H = nD / H  # new deaths per hospitalization

            y_accum.append(y)

        r_T_I = (
            self.testing_rate_f(t) / I
        )  # Test rate divided by infected not yet found (new, left overs)
        pos = self.model.positivity(r_T_I)  # Assumed (TODO fit) test positivity that will result
        exp_growth_factor = math.exp(
            (self.rt_f(self.t_list[-1]) - 1.0) / self.model.serial_period
        )  # Expected growth factor (in steady state) given injected R(t)

        # TODO pivot results to return
        return (
            y_accum,
            {
                "growth_factor": round(fractional_change, 3),
                "exp_growth_factor": round(exp_growth_factor, 3),
                "r_dD_nC": round(r_dD_nC, 4),
                "r_C_IC": round(r_C_IC, 2),
                "r_C_WIC": round(r_C_WIC, 2),
                "r_H_IC": round(r_H_IC, 2),
                "r_dD_H": round(r_dD_H, 3),
                "r_T_I": round(r_T_I, 2),
                "pos": round(pos, 3),
            },
        )

    def _time_step(self, y, t, dt=1.0, implicit_infections=False):
        """
        One integral moment. Included features beyond basic explicit integration forward in time
        of coupled ordinary differential equations with constant coefficients:
        0) It is possible to specify R(t) and use it adjust beta (infectivity) nonlinearly in each
           time step to ensure that new cases will grow as specified.
        1) When R(t) and implicit_infections are specified the C model compartment is not
           determined by integration - instead it is directly determined from R(t). Additionally,
           I is derived directly from the relationship between positivity and T/I while E and W
           are thereafter derived from the ODE rate equations.
        2) An additional delay in the input of values C, W, I to hospitalizations are also supported
           to patch up the unrealistic delay in ODE formulations of SEIR (verified needed but not yet tested)
        3) An approximate treatment of demographics is included that leverages input trends to adjust
           f_h (fraction of C,W,I that lead to hospitalizations) and f_d (fraction of hospitalizations
           that lead to deaths) using documented values by age segment
        4) Additional (non physical for "fitting") variations in the conversion of C/W/I to H and H to D
           are supported in the form of linear ramp coefficients (not yet tested)
        """
        (S, E, I, W, nC, C, H, D, R, b) = y
        model = self.model
        t_e = model.t_e
        t_i = model.t_i
        t_h = model.t_h

        k = (self.rt_f(t) - 1.0) / model.serial_period
        if implicit_infections:
            k_expected = math.exp(k)  # captures exponential growth during timestep
        else:
            k_expected = math.exp(k) - 1.0

        # transition fractions and linear ramp corrections
        if self.case_median_age_f is not None:
            median_age = self.case_median_age_f(t)
            fh0 = model.fh0_f(median_age=median_age)
            fd0 = model.fd0_f(median_age=median_age)
        else:
            fh0 = model.fh0_f()
            fd0 = model.fd0_f()
        fh = fh0 * model.lr_fh[0] + model.lr_fh[1] * (t - self.t_list[0])
        fh = max(0.0, min(1.0, fh))
        fd = fd0 * model.lr_fd[0] + model.lr_fd[1] * (t - self.t_list[0])
        fd = max(0.0, min(1.0, fd))
        fw = model.fw0  # might do something smarter later

        # Use delayed Cases and Infected to drive hospitalizations
        avg_R = self.rt_f(t)  # do better by averaging in the future
        growth_over_delay = math.exp((avg_R - 1.0) / model.serial_period * model.delay_ci_h)
        delayed_C = C / growth_over_delay
        delayed_W = W / growth_over_delay
        delayed_I = I / growth_over_delay

        if implicit_infections:  # E,I,W,C are determined directly from C(t-1), R(t) and positivity
            # C(t), nC determined directly from C(t-1) and R(t)
            positive_tests = k_expected * nC
            dCdt = positive_tests - delayed_C / t_i
            C_new = C + dCdt * dt

            # Which allows T/I to be inferred by inverting the positivity function
            # which determines I and dIdt
            pos_new = positive_tests / self.testing_rate_f(t)
            T_over_I = model.positivity_to_t_over_i(pos_new)
            I_new = self.testing_rate_f(t) / T_over_I
            dIdt = (I_new - I) / dt

            # Which allows E to be inferred and hence dEdt, number_exposed
            E_new = max(0.0, (dIdt + positive_tests + delayed_I / t_i) * t_e / (1.0 - fw))
            dEdt = (E_new - E) / dt
            number_exposed = dEdt + E / t_e

            # Which determines W
            dWdt = fw * E / t_e - delayed_W / t_i
            W_new = W + dWdt * dt

            # And finally number_exposed and hence beta number_exposed = beta *
            beta = number_exposed / (S * (I_new + W_new + C_new) / self.N)

        else:  # Calculate E,W,C,I explicitly forward in time
            beta = (k_expected + 1.0 / model.t_i) * (k_expected + 1.0 / t_e) * t_e
            number_exposed = beta * S * (I + W + C) / self.N

            tests_performed = self.testing_rate_f(t) * dt
            positive_tests = tests_performed * model.positivity(
                self.testing_rate_f(t) / max(I, 1.0)
            )

            dEdt = number_exposed - E / t_e
            dCdt = positive_tests - delayed_C / t_i
            dIdt = (1.0 - fw) * E / t_e - positive_tests - delayed_I / t_i
            dWdt = fw * E / t_e - delayed_W / t_i

        dSdt = -number_exposed

        dHdt = fh * (delayed_I + delayed_C) / t_i - H / t_h
        dDdt = fd * H / t_h
        if implicit_infections:
            dRdt = -(dSdt + dEdt + dIdt + dWdt + dCdt + dHdt + dDdt)
        else:
            dRdt = (1 - fh) * (delayed_I + delayed_C) / t_i + delayed_W / t_i + (1 - fd) * H / t_h

        # Validate conservation
        delta = dSdt + dEdt + dIdt + dWdt + dCdt + dHdt + dDdt + dRdt
        if delta > 1.0:
            assert True

        return (
            dt * dSdt,
            dt * dEdt,
            dt * dIdt,
            dt * dWdt,
            positive_tests,
            dt * dCdt,
            dt * dHdt,
            dt * dDdt,
            dt * dRdt,
            beta,
        )

    def plot_results(self, y_scale="log", xlim=None) -> plt.Figure:
        """
        Generate a summary plot for the simulation.

        Parameters
        ----------
        y_scale: str
            Matplotlib scale to use on y-axis. Typically 'log' or 'linear'
        """

        all_infected = self.results["A"] + self.results["I"] + self.results["C"]

        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig = plt.figure(facecolor="w", figsize=(20, 6))

        # ---------------------------- Left plot -------------------------
        plt.subplot(131)
        plt.plot(
            self.t_list,
            self.results["E"] / 100.0,
            alpha=0.5,
            lw=2,
            label="Exposed/100.",
            linestyle="--",
        )

        plt.plot(self.t_list, all_infected / 100.0, alpha=0.5, lw=2, label="Infected (A+I+C)/100.")
        h_all = self.results["H"]
        plt.plot(self.t_list, h_all / 10.0, alpha=0.5, lw=4, label="All Hospitalized/10.")
        plt.plot(self.t_list, self.results["nD"], alpha=0.5, lw=4, label="New Deaths")

        plt.xlabel("Time [days]", fontsize=12)
        plt.yscale(y_scale)
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(framealpha=0.5)

        if self.historical_compartments is not None:
            hc = self.historical_compartments
            plt.scatter(hc["H"].index, hc["H"].values / 10.0, c="g", marker=".")
            plt.scatter(hc["nD"].index, hc["nD"].values, c="r", marker=".")

        if xlim:
            plt.xlim(*xlim)
        # plt.gca().set_ylim(bottom=1)

        # ---------------------------- Center plot -------------------------
        plt.subplot(132)
        plt.plot(
            self.t_list,
            np.array([self.rt_f(t) / 10 for t in self.t_list]),
            alpha=0.5,
            lw=2,
            label="R(t)/10",
            linestyle="--",
        )
        plt.plot(
            self.t_list,
            self.results["I"] / np.array([self.testing_rate_f(t) for t in self.t_list]),
            alpha=0.5,
            lw=2,
            label="I/T",
        )
        plt.plot(
            self.t_list,
            np.array([self.model.fh0_f(median_age=self.case_median_age_f(t)) for t in self.t_list]),
            alpha=0.5,
            lw=4,
            label="fh(age)",
        )
        plt.plot(
            self.t_list,
            np.array([self.model.fd0_f(median_age=self.case_median_age_f(t)) for t in self.t_list]),
            alpha=0.5,
            lw=4,
            label="fd(age)",
        )
        plt.plot(
            self.t_list,
            np.array([self.testing_rate_f(t) / 100000 for t in self.t_list]),
            alpha=0.5,
            lw=2,
            label="Tests/100k",
            linestyle="--",
        )

        plt.xlabel("Time [days]", fontsize=12)
        plt.yscale(y_scale)
        plt.ylabel("")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(framealpha=0.5)
        if xlim:
            plt.xlim(*xlim)

        # ---------------------------- Right plot -------------------------
        # Reproduction numbers
        plt.subplot(133)

        plt.plot(
            self.t_list, self.results["b"], alpha=0.5, lw=2, label="beta", linestyle="--",
        )
        plt.plot(self.t_list, self.results["C"] / all_infected, alpha=0.5, lw=2, label="C/(I+C+A)")
        plt.plot(
            self.t_list, self.results["H"] / self.results["nC"], alpha=0.5, lw=4, label="H/nC",
        )
        plt.plot(
            self.t_list,
            10.0 * self.results["nD"] / self.results["nC"],
            alpha=0.5,
            lw=4,
            label="10*nD/nC",
        )
        plt.ylabel("Ratios check")
        plt.legend(framealpha=0.5)
        plt.yscale("log")
        plt.xlabel("Time [days]", fontsize=12)
        plt.grid(True, which="both")
        return fig


class SEIRModel:
    """
    This class implements a SEIR-like compartmental epidemic model
    consisting of SEIR states plus death, and hospitalizations.

    In the diff eq modeling, these parameters are assumed exponentially
    distributed and modeling occurs in the thermodynamic limit, i.e. we do
    not perform Monte Carlo for individual cases.

    Model Refs:
     # Dynamics have been verified against the SEIR plus package:
     # https://github.com/ryansmcgee/seirsplus#usage-install


     - https://arxiv.org/pdf/2003.10047.pdf  # We mostly follow this notation.
     - https://arxiv.org/pdf/2002.06563.pdf

    TODO: County-by-county affinity matrix terms can be used to describe
    transmission network effects. ( also known as Multi-Region SEIR)
    https://arxiv.org/pdf/2003.09875.pdf
     For those living in county i, the interacting county j exposure is given
     by A term dE_i/dt += N_i * Sum_j [ beta_j * mix_ij * I_j * S_i + beta_i *
     mix_ji * I_j * S_i ] mix_ij can be proxied by Census-based commuting
     matrices as workplace interactions are the dominant term. See:
     https://www.census.gov/topics/employment/commuting/guidance/flows.html

    TODO: Age-based contact mixing affinities.
          Incorporate structures from Weitz group
     - https://github.com/jsweitz/covid-19-ga-summer-2020/blob/master/fignearterm_0328_alt.m

       It is important to track demographics themselves as they impact
       hospitalization and mortality rates. Additionally, exposure rates vary
       by age, described by matrices linked below which need to be extracted
       from R for the US.
       https://cran.r-project.org/web/packages/socialmixr/vignettes/introduction.html
       For an infected age PMF vector I, and a contact matrix gamma dE_i/dT =
       S_i (*) gamma_ij I^j / N - gamma * E_i   # Someone should double check
       this

    Parameters
    ----------
    N: int
        Total population
    t_list: array-like
        Array of timesteps. Usually these are spaced daily.
    suppression_policy: callable
        Suppression_policy(t) should return a scalar in [0, 1] which
        represents the contact rate reduction from social distancing.
    A_initial: int
        Initial asymptomatic
    I_initial: int
        Initial infections.
    R_initial: int
        Initial recovered.
    E_initial: int
        Initial exposed
    HGen_initial: int
        Initial number of General hospital admissions.
    HICU_initial: int
        Initial number of ICU cases.
    HICUVent_initial: int
        Initial number of ICU cases.
    D_initial: int
        Initial number of deaths
    n_days: int
        Number of days to simulate.
    R0: float
        Basic Reproduction number
    R0_hospital: float
        Basic Reproduction number in the hospital.
    kappa: float
        Fractional contact rate for those with symptoms since they should be
        isolated vs asymptomatic who are less isolated. A value 1 implies
        the same rate. A value 0 implies symptomatic people never infect
        others.
    sigma: float
        Latent decay scale is defined as 1 / incubation period.
        1 / 4.8: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf
        1 / 5.2 [3, 8]: https://arxiv.org/pdf/2003.10047.pdf
    delta: float
        Infectious period
        See ICL report 13 for serial interval. We model infectious period as
        a bit longer with a Gamma(5, 1) which has a mean of 5
    delta_hospital: float
        Infectious period for patients in the hospital which is usually a bit
        longer.
    gamma: float
        Clinical outbreak rate (fraction of infected that show symptoms)
    hospitalization_rate_general: float
        Fraction of infected that are hospitalized generally (not in ICU)
    hospitalization_rate_icu: float
        Fraction of infected that are hospitalized in the ICU
    hospitalization_length_of_stay_icu_and_ventilator: float
        Mean LOS for those requiring ventilators
    fraction_icu_requiring_ventilator: float
        Of the ICU cases, which require ventilators.
    mortality_rate: float
        Fraction of infected that die.
        0.0052: https://arxiv.org/abs/2003.10720
        0.01
    beds_general: int
        General (non-ICU) hospital beds available.
    beds_ICU: int
        ICU beds available
    ventilators: int
        Ventilators available.
    symptoms_to_hospital_days: float
        Mean number of days elapsing between infection and
        hospital admission.
    hospitalization_length_of_stay_general: float
        Mean number of days for a hospitalized individual to be discharged.
    hospitalization_length_of_stay_icu
        Mean number of days for a ICU hospitalized individual to be
        discharged.
    mortality_rate_no_ICU_beds: float
        The percentage of those requiring ICU that die if ICU beds are not
        available.
    mortality_rate_no_ventilator: float
        The percentage of those requiring ventilators that die if they are
        not available.
    mortality_rate_no_general_beds: float
        The percentage of those requiring general hospital beds that die if
        they are not available.
    initial_hospital_bed_utilization: float
        Starting utilization fraction for hospital beds and ICU beds.
    hospital_capacity_change_daily_rate: float
        Rate of change (geometric increase in hospital bed capacity.
    max_hospital_capacity_factor: float
        Cap the hospital capacity.
    """

    def __init__(
        self,
        N,
        t_list,
        suppression_policy,
        A_initial=1,
        I_initial=1,
        R_initial=0,
        E_initial=0,
        HGen_initial=0,
        HICU_initial=0,
        HICUVent_initial=0,
        D_initial=0,
        R0=3.6,
        R0_hospital=0.6,
        sigma=1 / 3,  # -2 days because this is when contagious.
        delta=1 / 6,  # Infectious period
        delta_hospital=1 / 8,  # Infectious period
        kappa=1,
        gamma=0.5,
        hospitalization_rate_general=0.025,
        hospitalization_rate_icu=0.025,
        fraction_icu_requiring_ventilator=0.75,  # TBD Tuned...
        symptoms_to_hospital_days=5,
        hospitalization_length_of_stay_general=7,
        hospitalization_length_of_stay_icu=16,
        hospitalization_length_of_stay_icu_and_ventilator=17,
        beds_general=300,
        beds_ICU=100,
        ventilators=60,
        mortality_rate_from_ICU=0.4,
        mortality_rate_from_hospital=0.0,
        mortality_rate_no_ICU_beds=1.0,
        mortality_rate_from_ICUVent=1.0,
        mortality_rate_no_general_beds=0.0,
        initial_hospital_bed_utilization=0.6,
    ):

        self.N = N
        self.suppression_policy = suppression_policy
        self.I_initial = I_initial
        self.A_initial = A_initial
        self.R_initial = R_initial
        self.E_initial = E_initial
        self.D_initial = D_initial

        self.HGen_initial = HGen_initial
        self.HICU_initial = HICU_initial
        self.HICUVent_initial = HICUVent_initial

        self.S_initial = (
            self.N
            - self.A_initial
            - self.I_initial
            - self.R_initial
            - self.E_initial
            - self.D_initial
            - self.HGen_initial
            - self.HICU_initial
            - self.HICUVent_initial
        )

        # Epidemiological Parameters
        self.R0 = R0  # Reproduction Number
        self.R0_hospital = R0_hospital  # Reproduction Number
        self.sigma = sigma  # 1 / Incubation period
        self.delta = delta  # 1 / Infectious period
        self.delta_hospital = delta_hospital  # 1 / Infectious period
        self.gamma = gamma  # Clinical outbreak rate for those infected.
        self.kappa = kappa  # Reduce contact due to isolation of symptomatic cases.

        # These need to be made age dependent R0 =  beta = Contact rate * infectious period.
        self.beta = self.R0 * self.delta
        self.beta_hospital = self.R0_hospital * self.delta_hospital

        self.symptoms_to_hospital_days = symptoms_to_hospital_days

        # Hospitalization Parameters
        # https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf
        # Page 16
        self.hospitalization_rate_general = hospitalization_rate_general
        self.hospitalization_rate_icu = hospitalization_rate_icu
        self.hospitalization_length_of_stay_general = hospitalization_length_of_stay_general
        self.hospitalization_length_of_stay_icu = hospitalization_length_of_stay_icu
        self.hospitalization_length_of_stay_icu_and_ventilator = (
            hospitalization_length_of_stay_icu_and_ventilator
        )

        # http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf
        # = 0.53
        self.fraction_icu_requiring_ventilator = fraction_icu_requiring_ventilator

        # Capacity
        self.beds_general = beds_general
        self.beds_ICU = beds_ICU
        self.ventilators = ventilators

        self.mortality_rate_no_general_beds = mortality_rate_no_general_beds
        self.mortality_rate_no_ICU_beds = mortality_rate_no_ICU_beds
        self.mortality_rate_from_ICUVent = mortality_rate_from_ICUVent
        self.initial_hospital_bed_utilization = initial_hospital_bed_utilization

        self.mortality_rate_from_ICU = mortality_rate_from_ICU
        self.mortality_rate_from_hospital = mortality_rate_from_hospital

        # List of times to integrate.
        self.t_list = t_list
        self.results = None

    def _time_step(self, y, t):
        """
        One integral moment.

        y: array
            S, E, A, I, R, HNonICU, HICU, HICUVent, D = y
        """
        (
            S,
            E,
            A,
            I,
            R,
            HNonICU,
            HICU,
            HICUVent,
            D,
            dHAdmissions_general,
            dHAdmissions_icu,
            dTotalInfections,
        ) = y

        # Effective contact rate * those that get exposed * those susceptible.
        number_exposed = (
            self.beta * self.suppression_policy(t) * S * (self.kappa * I + A) / self.N
            + self.beta_hospital * S * (HICU + HNonICU) / self.N
        )
        dSdt = -number_exposed

        exposed_and_symptomatic = (
            self.gamma * self.sigma * E
        )  # latent period moving to infection = 1 / incubation
        exposed_and_asymptomatic = (
            (1 - self.gamma) * self.sigma * E
        )  # latent period moving to asymptomatic but infected) = 1 / incubation
        dEdt = number_exposed - exposed_and_symptomatic - exposed_and_asymptomatic

        asymptomatic_and_recovered = self.delta * A
        dAdt = exposed_and_asymptomatic - asymptomatic_and_recovered

        # Fraction that didn't die or go to hospital
        infected_and_recovered_no_hospital = self.delta * I
        infected_and_in_hospital_general = (
            I
            * (self.hospitalization_rate_general - self.hospitalization_rate_icu)
            / self.symptoms_to_hospital_days
        )
        infected_and_in_hospital_icu = (
            I * self.hospitalization_rate_icu / self.symptoms_to_hospital_days
        )

        dIdt = (
            exposed_and_symptomatic
            - infected_and_recovered_no_hospital
            - infected_and_in_hospital_general
            - infected_and_in_hospital_icu
        )

        mortality_rate_ICU = (
            self.mortality_rate_from_ICU
            if HICU <= self.beds_ICU
            else self.mortality_rate_no_ICU_beds
        )
        mortality_rate_NonICU = (
            self.mortality_rate_from_hospital
            if HNonICU <= self.beds_general
            else self.mortality_rate_no_general_beds
        )

        died_from_hosp = (
            HNonICU * mortality_rate_NonICU / self.hospitalization_length_of_stay_general
        )
        died_from_icu = (
            HICU
            * (1 - self.fraction_icu_requiring_ventilator)
            * mortality_rate_ICU
            / self.hospitalization_length_of_stay_icu
        )
        died_from_icu_vent = (
            HICUVent
            * self.mortality_rate_from_ICUVent
            / self.hospitalization_length_of_stay_icu_and_ventilator
        )

        recovered_after_hospital_general = (
            HNonICU * (1 - mortality_rate_NonICU) / self.hospitalization_length_of_stay_general
        )
        recovered_from_icu_no_vent = (
            HICU
            * (1 - mortality_rate_ICU)
            * (1 - self.fraction_icu_requiring_ventilator)
            / self.hospitalization_length_of_stay_icu
        )
        recovered_from_icu_vent = (
            HICUVent
            * (1 - max(mortality_rate_ICU, self.mortality_rate_from_ICUVent))
            / self.hospitalization_length_of_stay_icu_and_ventilator
        )

        dHNonICU_dt = (
            infected_and_in_hospital_general - recovered_after_hospital_general - died_from_hosp
        )
        dHICU_dt = (
            infected_and_in_hospital_icu
            - recovered_from_icu_no_vent
            - recovered_from_icu_vent
            - died_from_icu
            - died_from_icu_vent
        )

        # This compartment is for tracking ventilator count. The beds are
        # accounted for in the ICU cases.
        dHICUVent_dt = (
            infected_and_in_hospital_icu * self.fraction_icu_requiring_ventilator
            - HICUVent / self.hospitalization_length_of_stay_icu_and_ventilator
        )

        # Tracking categories...
        dTotalInfections = exposed_and_symptomatic + exposed_and_asymptomatic
        dHAdmissions_general = infected_and_in_hospital_general
        dHAdmissions_ICU = infected_and_in_hospital_icu  # Ventilators also count as ICU beds.

        # Fraction that recover
        dRdt = (
            asymptomatic_and_recovered
            + infected_and_recovered_no_hospital
            + recovered_after_hospital_general
            + recovered_from_icu_vent
            + recovered_from_icu_no_vent
        )

        # TODO Age dep mortality. Recent estimate fo relative distribution Fig 3 here:
        #      http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf
        dDdt = died_from_icu + died_from_icu_vent + died_from_hosp  # Fraction that die.

        return (
            dSdt,
            dEdt,
            dAdt,
            dIdt,
            dRdt,
            dHNonICU_dt,
            dHICU_dt,
            dHICUVent_dt,
            dDdt,
            dHAdmissions_general,
            dHAdmissions_ICU,
            dTotalInfections,
        )

    def run(self):
        """
        Integrate the ODE numerically.

        Returns
        -------
        results: dict
        {
            't_list': self.t_list,
            'S': S,
            'E': E,
            'I': I,
            'R': R,
            'HNonICU': HNonICU,
            'HICU': HICU,
            'HVent': HVent,
            'D': Deaths from straight mortality. Not including hospital saturation deaths,
            'deaths_from_hospital_bed_limits':
            'deaths_from_icu_bed_limits':
            'deaths_from_ventilator_limits':
            'total_deaths':
        }
        """
        # Initial conditions vector
        HAdmissions_general, HAdmissions_ICU, TotalAllInfections = 0, 0, 0
        y0 = (
            self.S_initial,
            self.E_initial,
            self.A_initial,
            self.I_initial,
            self.R_initial,
            self.HGen_initial,
            self.HICU_initial,
            self.HICUVent_initial,
            self.D_initial,
            HAdmissions_general,
            HAdmissions_ICU,
            TotalAllInfections,
        )

        # Integrate the SEIR equations over the time grid, t.
        result_time_series = odeint(self._time_step, y0, self.t_list, atol=1e-3, rtol=1e-3)
        (
            S,
            E,
            A,
            I,
            R,
            HGen,
            HICU,
            HICUVent,
            D,
            HAdmissions_general,
            HAdmissions_ICU,
            TotalAllInfections,
        ) = result_time_series.T

        self.results = {
            "t_list": self.t_list,
            "S": S,
            "E": E,
            "A": A,
            "I": I,
            "R": R,
            "HGen": HGen,
            "HICU": HICU,
            "HVent": HICUVent,
            "D": D,
            "direct_deaths_per_day": derivative(D),  # Derivative...
            # Here we assume that the number of person days above the saturation
            # divided by the mean length of stay approximates the number of
            # deaths from each source.
            "deaths_from_hospital_bed_limits": np.cumsum((HGen - self.beds_general).clip(min=0))
            * self.mortality_rate_no_general_beds
            / self.hospitalization_length_of_stay_general,
            # Here ICU = ICU + ICUVent, but we want to remove the ventilated
            # fraction and account for that below.
            "deaths_from_icu_bed_limits": np.cumsum((HICU - self.beds_ICU).clip(min=0))
            * self.mortality_rate_no_ICU_beds
            / self.hospitalization_length_of_stay_icu,
            "HGen_cumulative": np.cumsum(HGen) / self.hospitalization_length_of_stay_general,
            "HICU_cumulative": np.cumsum(HICU) / self.hospitalization_length_of_stay_icu,
            "HVent_cumulative": np.cumsum(HICUVent)
            / self.hospitalization_length_of_stay_icu_and_ventilator,
        }

        self.results["total_deaths"] = D

        # Derivatives of the cumulative give the "new" infections per day.
        self.results["total_new_infections"] = derivative(TotalAllInfections)
        self.results["total_deaths_per_day"] = derivative(self.results["total_deaths"])
        self.results["general_admissions_per_day"] = derivative(HAdmissions_general)
        self.results["icu_admissions_per_day"] = derivative(
            HAdmissions_ICU
        )  # Derivative of the cumulative.

    def plot_results(self, y_scale="log", xlim=None, alternate_plots=False) -> plt.Figure:
        """
        Generate a summary plot for the simulation.

        Parameters
        ----------
        y_scale: str
            Matplotlib scale to use on y-axis. Typically 'log' or 'linear'
        """
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig = plt.figure(facecolor="w", figsize=(20, 6))

        # ---------------------------- Left plot -------------------------
        plt.subplot(131)
        if not alternate_plots:
            plt.plot(self.t_list, self.results["S"], alpha=1, lw=2, label="Susceptible")
        plt.plot(self.t_list, self.results["E"], alpha=0.5, lw=2, label="Exposed", linestyle="--")

        if not alternate_plots:
            plt.plot(self.t_list, self.results["A"], alpha=0.5, lw=2, label="Asymptomatic")
            plt.plot(self.t_list, self.results["I"], alpha=0.5, lw=2, label="Infected")
            plt.plot(
                self.t_list,
                self.results["R"],
                alpha=1,
                lw=2,
                label="Recovered & Immune",
                linestyle="--",
            )
            plt.plot(
                self.t_list,
                self.results["S"]
                + self.results["E"]
                + self.results["A"]
                + self.results["I"]
                + self.results["R"]
                + self.results["D"]
                + self.results["HGen"]
                + self.results["HICU"],
                label="Total",
            )
        else:
            a_plus_i = self.results["A"] + self.results["I"]
            plt.plot(self.t_list, a_plus_i, alpha=0.5, lw=2, label="Infected + Asymptomatic")
            h_all = self.results["HGen"] + self.results["HICU"] + self.results["HVent"]
            plt.plot(self.t_list, h_all, alpha=0.5, lw=4, label="All Hospitalized")
            plt.plot(
                self.t_list,
                self.results["direct_deaths_per_day"],
                alpha=0.5,
                lw=4,
                label="New Deaths",
            )

        plt.xlabel("Time [days]", fontsize=12)
        plt.yscale(y_scale)
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(framealpha=0.5)
        if xlim:
            plt.xlim(*xlim)
        else:
            plt.xlim(0, self.t_list.max())
        if not alternate_plots:
            plt.ylim(1, self.N * 1.1)
        else:
            plt.gca().set_ylim(bottom=1)

        # ---------------------------- Center plot -------------------------
        plt.subplot(132)

        plt.plot(
            self.t_list,
            self.results["D"],
            alpha=0.4,
            c="k",
            lw=1,
            label="Direct Deaths",
            linestyle="-",
        )
        plt.plot(
            self.t_list,
            self.results["deaths_from_hospital_bed_limits"],
            alpha=1,
            c="k",
            lw=1,
            label="Deaths From Bed Limits",
            linestyle=":",
        )
        plt.plot(
            self.t_list,
            self.results["deaths_from_icu_bed_limits"],
            alpha=1,
            c="k",
            lw=2,
            label="Deaths From ICU Bed Limits",
            linestyle="-.",
        )
        plt.plot(
            self.t_list,
            self.results["total_deaths"],
            alpha=1,
            c="k",
            lw=4,
            label="Total Deaths",
            linestyle="-",
        )

        plt.plot(
            self.t_list,
            self.results["HGen"],
            alpha=1,
            lw=2,
            c="steelblue",
            label="General Beds Required",
            linestyle="-",
        )
        plt.hlines(
            self.beds_general,
            self.t_list[0],
            self.t_list[-1],
            "steelblue",
            alpha=1,
            lw=2,
            label="General Bed Capacity",
            linestyle="--",
        )

        plt.plot(
            self.t_list,
            self.results["HICU"],
            alpha=1,
            lw=2,
            c="firebrick",
            label="ICU Beds Required",
            linestyle="-",
        )
        plt.hlines(
            self.beds_ICU,
            self.t_list[0],
            self.t_list[-1],
            "firebrick",
            alpha=1,
            lw=2,
            label="ICU Bed Capacity",
            linestyle="--",
        )

        plt.plot(
            self.t_list,
            self.results["HVent"],
            alpha=1,
            lw=2,
            c="seagreen",
            label="Ventilators Required",
            linestyle="-",
        )
        plt.hlines(
            self.ventilators,
            self.t_list[0],
            self.t_list[-1],
            "seagreen",
            alpha=1,
            lw=2,
            label="Ventilator Capacity",
            linestyle="--",
        )

        plt.xlabel("Time [days]", fontsize=12)
        plt.ylabel("")
        plt.yscale(y_scale)
        plt.ylim(1, plt.ylim()[1])
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(framealpha=0.5)
        if xlim:
            plt.xlim(*xlim)
        else:
            plt.xlim(0, self.t_list.max())

        # ---------------------------- Right plot -------------------------
        # Reproduction numbers
        plt.subplot(133)

        if not alternate_plots:
            plt.plot(self.t_list, [self.suppression_policy(t) for t in self.t_list], c="steelblue")
            plt.ylabel("Contact Rate Reduction")
        else:
            plt.plot(
                self.t_list, self.results["E"] / self.results["I"], alpha=0.5, lw=2, label="E/I"
            )
            plt.plot(
                self.t_list, self.results["A"] / self.results["I"], alpha=0.5, lw=2, label="A/I"
            )
            plt.plot(
                self.t_list,
                (self.results["HGen"] + self.results["HICU"] + self.results["HVent"])
                / self.results["I"],
                alpha=0.5,
                lw=4,
                label="H/I",
            )
            plt.plot(
                self.t_list,
                self.results["direct_deaths_per_day"]
                / (self.results["HGen"] + self.results["HICU"] + self.results["HVent"]),
                alpha=0.5,
                lw=4,
                label="D/H",
            )
            plt.ylabel("Ratios check")
            plt.legend(framealpha=0.5)
            plt.yscale("log")
        plt.xlabel("Time [days]", fontsize=12)
        plt.grid(True, which="both")
        return fig
