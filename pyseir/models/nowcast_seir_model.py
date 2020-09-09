import numpy as np
import pandas as pd
import math
from enum import Enum
import copy

import matplotlib.pyplot as plt

from pyseir.models.demographics import Demographics


def ramp_function(t_list, start_value, end_value):
    """
    Returns a function that implements a linear ramp from start_value to end_value over
    the time domain present in (an ordered) t_list
    """
    rate = (end_value - start_value) / (t_list[-1] - t_list[0])
    ftn = lambda t: start_value + (t - t_list[0]) * rate
    return ftn


def extend_rt_function_with_new_cases_forecast(rt_f, serial_period, forecast_new_cases):
    """
    Takes an rt_f function and extends it to match up with one or more
    ordered new_case forecasts in the future (after today). Note the first forecast
    must be a specification of the new_cases today (the start of the forecast period)
    """
    using = copy.deepcopy(forecast_new_cases)
    (today, current_nC) = using.pop(0)
    r_ftn = rt_f

    def make_switching_ftn(f, m, today):
        def ftn(t):
            if t <= today:
                return f(t)
            else:
                return f(today) + m * (t - today)

        return ftn

    while len(using) > 0:
        (future, target_nC) = using.pop(0)
        assert future > today
        nC_ratio = target_nC / current_nC

        # Assuming rt is of form rt(t) = rt(today) + m * (t-today)
        # where also Integral of (rt(t) - 1)/serial_period from t=today->future is ln(nC_ratio)
        # so the predicted nC value will match up at t=future
        r0 = r_ftn(today)
        const_integral = 1.0 / serial_period * (r0 - 1) * (future - today)
        variable_integral = math.log(nC_ratio) - const_integral
        m = variable_integral / (0.5 * (future - today) ** 2) * serial_period

        r_ftn = make_switching_ftn(r_ftn, m, today)

        # Get ready to iterate to the next forecast if needed
        (today, current_nC) = (future, target_nC)

    return r_ftn


# Transition fractions to Hospitalizations and Deaths by decade
# to age groups: 0-35, 35-65 and 65-100
# TODO need to normalize contributions per decade by local demographics

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

# TODO still decide which is more appropriate for US
FH = FH_BY_DECADE_BC
CFR = CFR_BY_DECADE_BC  # CFR_BY_2DECADE_IA

FH_BY_AGE = [
    (FH[0] + FH[1] + FH[2] + 0.5 * FH[3]) / 3.5,
    (0.5 * FH[3] + FH[4] + FH[5] + 0.5 * FH[6]) / 3,
    (0.5 * FH[6] + FH[7] + FH[8] + FH[9]) / 3.5,
]
if len(CFR) > 5:  # by decade
    CFR_BY_AGE = [
        (CFR[0] + CFR[1] + CFR[2] + 0.5 * CFR[3]) / 3.5,
        (0.5 * CFR[3] + CFR[4] + CFR[5] + 0.5 * CFR[6]) / 3,
        (0.5 * CFR[6] + CFR[7] + CFR[8] + CFR[9]) / 3.5,
    ]
else:  # by 2 decades
    CFR_BY_AGE = [
        (CFR[0] + 0.75 * CFR[1]) / 1.75,
        (0.25 * CFR[1] + CFR[2] + 0.25 * CFR[3]) / 1.5,
        (0.75 * CFR[3] + CFR[4]) / 1.75,
    ]
FD_BY_AGE = [CFR_BY_AGE[i] / FH_BY_AGE[i] for i in range(0, 3)]


class NowcastingSEIRModel:
    """
    Simplified SEIR Model sheds complexity where not needed to be accurate enough. See ModelRun and 
    ModelRun._time_step for details
    
    The concept of running the Model has been split off into another class (ModelRun).

    TODO Next steps:
    * Turn on delay and see if that helps match peak behaviour
    * Validate test positivity contribution to peaks against that observed
    """

    FRACTION_HOSP_DWELL_AS_DELAY = 1.0

    def __init__(
        self,
        # ____________These are (to be) trained factors that are passed in___________
        median_age=None,
        lr_fh=(1.0, 0.0),
        lr_fd=(1.0, 0.0),
        delay_ci_h=0,  # added days of delay between infection and hospitalization
        age_eval_delay=None,  # 14,
    ):
        # Retain passed in parameters
        self.lr_fh = lr_fh
        self.lr_fd = lr_fd
        self.delay_ci_h = delay_ci_h  # not using this yet
        self.age_eval_delay = age_eval_delay

        # __________Fixed parameters not trained_________
        self.t_e = 2.0  # 2.0  sigma was 1/3. days^-1 below
        # TODO serial period as ~5 days in our old model runs. How is this not 6 = 3 + 1/2 * 6?
        self.t_i = 6.0  # delta was 1/6. days^-1 below
        self.serial_period = self.t_e + 0.5 * self.t_i  # this is 5 days agrees with delta below

        # Hospitalization stay length is important driver of total Hospitalizations (due to accumulation)
        # Range of reported values from 5-8 days. Some examples
        # https://www.medrxiv.org/content/medrxiv/early/2020/05/05/2020.04.30.20084780.full.pdf
        self.th0 = 6.0

        def th_f(med_age=None):
            if med_age is None:
                return self.th0
            else:
                return self.th0 * (1.0 + 0.5 * max(1.0, (med_age - 30.0) / 30.0))

        self.t_h = th_f

        if self.age_eval_delay is None:
            self.age_eval_delay = (
                self.t_i + NowcastingSEIRModel.FRACTION_HOSP_DWELL_AS_DELAY * self.t_h()
            )
        self.fw0 = 0.5
        self.pos0 = 0.5  # max positivity for testing -> 0
        self.pos_c = 1.75
        self.pos_x0 = 2.0
        # Solution that ensures continuity of function below and its derivative
        self.pos_b = (3.0 * self.pos_x0 - self.pos_c) / (4.0 * self.pos_x0 ** 1.5)
        self.pos_d = self.pos_x0 ** 0.5 / 4.0 * (3.0 * self.pos_c - self.pos_x0)

    def get_calibration(self):
        """
        Get the current calibration which is only using one proportionality factor for each of
        fh, fd
        """
        return (self.lr_fh[0], self.lr_fd[0])

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
                t_list=np.linspace(0, 100, 100 + 1),
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
            fractions = [f_young, (1.0 - f_young - f_old), f_old]
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
    * Starting from bare minimum of values: C or I
    * Starting from just a few observables for today: C, D, nC, H, nD (last 3 as constraints)
    * Starting from DataFrame outputs of a previous run

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
                case_median_age_f(t0 - model.age_eval_delay),
                t_over_x=max(testing_rate_f(t0) / x, 2.2) if testing_rate_f is not None else 100.0,
                x_is_new_cases=True if nC_initial is not None else False,
            )
            self.compartment_ratios_initial = compartments
        else:
            self.compartment_ratios_initial = compartment_ratios_initial  # get applied at run time

        S = S_initial if S_initial is not None else self.N
        self.history = ModelRun.make_array(
            S=S, I=I_initial, nC=nC_initial, H=H_initial, nD=nD_initial
        )

    def execute_dataframe_ratios_fig(self, plot=True):
        (history, ratios) = self.execute_lists_ratios()
        df = ModelRun.array_to_df(history)
        self.results = df
        if plot:
            fig = self.plot_results()
        else:
            fig = None
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
            # reset new recoveries and deaths
            y[8] = 0.0
            y[9] = 0.0

        if self.auto_calibrate:
            current = ModelRun.array_to_dict(y)
            adj_H = 1.0
            adj_nD = 1.0
            if y0["H"] > 0.0 and current["H"] > 0.0:
                adj_H = max(min(y0["H"] / current["H"], 10.0), 0.1)
                current["H"] = y0["H"]
            if y0["nD"] > 0.0 and current["nD"] > 0.0:
                adj_nD = max(min(y0["nD"] / current["nD"] / adj_H, 10.0), 0.1)
                current["nD"] = y0["nD"]
            y = ModelRun.dict_to_array(current)
            self.model.adjustFractions(adj_H, adj_nD)

        y_accum = list()
        y_accum.append(y)

        implicit = False if self.force_stationary else True

        # If today is set and is in the future than we are calibrating on more than one day
        if self.today is not None and self.today > self.t_list[0] + 7:  # days

            # Run without adjustments up until today, then compute adjustments needed to match today
            for t in np.linspace(self.t_list[0], today, int(today - self.t_list[0] + 1)):
                y = list(y)
                dy = self._time_step(y, t, dt=1.0, implicit_infections=implicit)
                (dS, dE, dI, dW, nC, dC, dH, dD, dR, b) = dy
                y_new = [max(0.1, a + b) for a, b in zip(y, dy)]
                # do not accumulate daily new cases, deaths or beta
                y_new[4] = nC
                y_new[7] = dD
                y_new[9] = b

                y = y_new
                y_accum.append(y)
            start_main = today

            # TODO calculate and apply adjustments
        else:
            start_main = self.t_list[0]

        # Iterate over all time steps
        remaining = np.linspace(start_main, self.t_list[-1], int(self.t_list[-1] - start_main + 1))

        if self.historical_compartments is not None:
            hc = self.historical_compartments
            smape_sum = 0.0
            smape_count = 0.0
            calculating_smape = True
        else:
            calculating_smape = False

        for t in remaining[:-1]:  # self.t_list[:-1]:
            y = list(y)
            dy = self._time_step(y, t, dt=1.0, implicit_infections=implicit)
            (dS, dE, dI, dW, nC, dC, dH, dD, dR, b) = dy
            y_new = [max(0.1, a + b) for a, b in zip(y, dy)]
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

            if calculating_smape:
                for c in ["H", "nD"]:
                    if (
                        hc[c] is not None  # have data
                        and t in hc[c].index  # for this time
                        and not math.isnan(hc[c][t])  # that is a number
                        and hc[c][t] > 0.3  # and we're not totally in "shot noise"
                    ):
                        val = H if c == "H" else nD
                        smape_sum += abs(val - hc[c][t]) / ((abs(val) + abs(hc[c][t])) / 2.0)
                        smape_count += 1
            y_accum.append(y)

        smape = smape_sum / smape_count if calculating_smape else 0.0
        r_T_I = (
            self.testing_rate_f(t) / I if self.testing_rate_f is not None else 100.0
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
                "SMAPE": round(smape, 3),
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

        med_age_h = self.case_median_age_f(t - int(model.age_eval_delay / 2))
        med_age_d = self.case_median_age_f(t - model.age_eval_delay)

        t_e = model.t_e
        t_i = model.t_i
        t_h = model.t_h(med_age_h)

        k = (self.rt_f(t) - 1.0) / model.serial_period
        if implicit_infections:
            k_expected = math.exp(k)  # captures exponential growth during timestep
        else:
            k_expected = math.exp(k) - 1.0

        # transition fractions and linear ramp corrections
        if self.case_median_age_f is not None:
            fh0 = model.fh0_f(median_age=med_age_h)
            fd0 = model.fd0_f(median_age=med_age_d)
        else:
            fh0 = model.fh0_f()
            fd0 = model.fd0_f()
        fh = fh0 * model.lr_fh[0] + model.lr_fh[1] * (t - self.t_list[0])
        fh = max(0.0, min(1.0, fh))
        fd = fd0 * model.lr_fd[0] + model.lr_fd[1] * (t - self.t_list[0])
        fd = max(0.0, min(1.0, fd))
        fw = model.fw0  # might do something smarter later

        # Damping factors for numerical stability
        e_max_growth = 2.0

        # Use delayed Cases and Infected to drive hospitalizations
        avg_R = self.rt_f(t)  # do better by averaging in the future
        growth_over_delay = math.exp((avg_R - 1.0) / model.serial_period * model.delay_ci_h)
        delayed_C = C / growth_over_delay
        delayed_W = W / growth_over_delay
        delayed_I = I / growth_over_delay

        # TODO this probably needs to be implicit to avoid instability which we are seeing
        # as thresholds in stiff function inv_positivity are crossed
        if implicit_infections:  # E,I,W,C are determined directly from C(t-1), R(t) and positivity
            # C(t), nC determined directly from C(t-1) and R(t)
            positive_tests = k_expected * nC
            dCdt = positive_tests - delayed_C / t_i
            C_new = C + dCdt * dt

            # Which allows T/I to be inferred by inverting the positivity function
            # which determines I and dIdt
            if self.testing_rate_f is not None:
                pos_new = min(0.4, positive_tests / self.testing_rate_f(t))
                T_over_I = model.positivity_to_t_over_i(pos_new)
                I_new = self.testing_rate_f(t) / T_over_I
            else:  # no testing function -> assume I -> 0 with only C and W
                pos_new = 0.01
                I_new = 1.0
            dIdt = (I_new - I) / dt

            # Which allows E to be inferred and hence dEdt, number_exposed
            E_new = max(10.0, (dIdt + positive_tests + delayed_I / t_i) * t_e / (1.0 - fw))
            E_new = max(
                min(e_max_growth * E, E_new), 1.0 / e_max_growth * E
            )  # Damp down rapid changes
            dEdt = (E_new - E) / dt
            number_exposed = dEdt + E / t_e

            # Which determines W
            dWdt = fw * E / t_e - delayed_W / t_i
            W_new = W + dWdt * dt

            # And finally number_exposed and hence beta number_exposed = beta *
            beta = max(0.01, number_exposed / (S * (I_new + W_new + C_new) / self.N))

        else:  # Calculate E,W,C,I explicitly forward in time
            beta = (k_expected + 1.0 / model.t_i) * (k_expected + 1.0 / t_e) * t_e
            number_exposed = beta * S * (I + W + C) / self.N

            tests_performed = self.testing_rate_f(t) * dt
            positive_tests = tests_performed * model.positivity(
                self.testing_rate_f(t) / max(I, 1.0)
            )
            # TODO unstable if positive_tests < nC with k>0

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

        low_ratio = [0.03 for t in self.t_list]

        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig = plt.figure(facecolor="w", figsize=(20, 6))
        fig.suptitle("Calibration: fh0=%.2f, fd0=%.2f" % (self.model.lr_fh[0], self.model.lr_fd[0]))

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

        plt.plot(self.t_list, self.results["nC"] / 10.0, alpha=0.5, lw=2, label="New Cases/10.")
        h_all = self.results["H"]
        plt.plot(self.t_list, h_all / 10.0, alpha=0.5, lw=4, label="All Hospitalized/10.")
        plt.plot(self.t_list, self.results["nD"], alpha=0.5, lw=4, label="New Deaths")

        plt.xlabel("Time [days]", fontsize=12)
        plt.yscale(y_scale)
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(framealpha=0.5)

        if self.historical_compartments is not None:
            hc = self.historical_compartments
            plt.scatter(hc["nC"].index, hc["nC"].values / 10.0, c="orange", marker=".")
            plt.scatter(hc["H"].index, hc["H"].values / 10.0, c="green", marker=".")
            plt.scatter(hc["nD"].index, hc["nD"].values, c="red", marker=".")

        # TODO set scale

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
        if self.testing_rate_f is not None:
            it_ratio = self.results["I"] / np.array([self.testing_rate_f(t) for t in self.t_list])
        else:
            it_ratio = low_ratio
        plt.plot(
            self.t_list, it_ratio, alpha=0.5, lw=2, label="I/T",
        )
        plt.plot(
            self.t_list,
            np.array(
                [
                    self.model.fh0_f(
                        median_age=self.case_median_age_f(t - int(self.model.age_eval_delay / 2))
                    )
                    for t in self.t_list
                ]
            ),
            alpha=0.5,
            lw=4,
            label="fh(age) delayed",
        )
        plt.plot(
            self.t_list,
            np.array(
                [
                    self.model.fd0_f(
                        median_age=self.case_median_age_f(t - self.model.age_eval_delay)
                    )
                    for t in self.t_list
                ]
            ),
            alpha=0.5,
            lw=4,
            label="fd(age) delayed",
        )
        if self.testing_rate_f is not None:
            plt.plot(
                self.t_list,
                np.array([self.testing_rate_f(t) / 100000 for t in self.t_list]),
                alpha=0.5,
                lw=2,
                label="Tests/100k",
                linestyle="--",
            )

        if self.historical_compartments is not None:
            hc = self.historical_compartments
            plt.scatter(
                hc["nD"].index,
                hc["nD"].values / hc["H"].values * self.model.t_h(),
                c="red",
                marker=".",
                label="fd actual",
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

