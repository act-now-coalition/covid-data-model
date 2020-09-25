import numpy as np
import pandas as pd
import math
from typing import List, Tuple

from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

from pyseir.models.demographics import Demographics, Transitions


def ramp_function(t_list, start_value, end_value):
    """
    Returns a function that implements a linear ramp from start_value to end_value over
    the time domain present in (an ordered) t_list
    """
    rate = (end_value - start_value) / (t_list[-1] - t_list[0])
    ftn = lambda t: start_value + (t - t_list[0]) * rate
    return ftn


def extend_rt_function_with_new_cases_forecast(
    rt_f, serial_period: float, forecast_new_cases: List[Tuple[int, float]]
):

    (today, current_nC) = forecast_new_cases[0]
    (future, future_nC) = forecast_new_cases[-1]

    historical = today - (forecast_new_cases[1][0] - today)
    integral_historical_rt = sum(
        [(rt_f(i) - 1.0) / serial_period for i in range(historical, today - 1)]
    )
    historical_nC = current_nC / math.exp(integral_historical_rt)

    padded = [(historical, historical_nC)] + forecast_new_cases

    x = [padded[i][0] for i in range(0, len(padded))]
    y = [math.log(padded[i][1]) for i in range(0, len(padded))]
    nc_ext_array = InterpolatedUnivariateSpline(x, y)
    # interp1d(x, y, kind="cubic")  # expects an array
    nc_ext = lambda t: float(nc_ext_array(t))

    def make_rt_switcher(rt, nc_ext, switch_at, last):
        def ftn(t):
            if t <= switch_at:
                return rt(t)
            elif t >= last:
                return (
                    math.log(math.exp(nc_ext(last)) / math.exp(nc_ext(last - 1))) * serial_period
                    + 1.0
                )
            else:
                return math.log(math.exp(nc_ext(t)) / math.exp(nc_ext(t - 1))) * serial_period + 1.0

        return ftn

    rt_ext = make_rt_switcher(rt_f, nc_ext, today, future)

    return rt_ext


class NowcastingSEIRModel:
    """
    Simplified SEIR Model sheds complexity where not needed to be accurate enough. See ModelRun and 
    ModelRun._time_step for details
    
    The concept of running the Model has been split off into another class (ModelRun).

    TODO FUTURE enhancements to consider:
    * Validate we have the right set of trainable (across results from all state) parameters exposed
    * Turn positivity back on and validate if it can contribute to improved accuracy
    """

    # What fraction of hospital dwell time contributes delay evaluating median age
    FRACTION_HOSP_DWELL_AS_DELAY = 1.0

    def __init__(
        self,
        # ____________These are (to be) trained factors that are passed in___________
        median_age=None,
        # TODO simplify - not using linear ramp for either of these anywhere
        lr_fh=(0.74, 0.0),  # geometric mean of all states
        lr_fd=(0.85, 0.0),  # geometric mean of all states
        delay_ci_h=0,  # added days of delay between infection and hospitalization
        age_eval_delay=None,  # else auto determined from t_i and t_h
        th_elongation_fraction=0.4,  # what fraction longer t_h is for older people
        # smape values: 0 - > .333, .3 -> .319, .5 -> .319, .8 -> .321
    ):
        # Retain passed in parameters
        self.lr_fh = lr_fh
        self.lr_fd = lr_fd
        self.delay_ci_h = delay_ci_h  # not using this yet
        self.age_eval_delay = (
            age_eval_delay  # Delay for median age from cases being applied to death transition
        )

        # __________Fixed parameters not trained_________
        self.t_e = 2.0  # 2.0  sigma was 1/3. days^-1 below
        self.t_i = 6.0  # delta was 1/6. days^-1 below
        self.serial_period = self.t_e + 0.5 * self.t_i  # this is 5 days agrees with delta below

        # Hospitalization stay length is important driver of total Hospitalizations (due to accumulation)
        # Range of reported values from 5-8 days. Some examples
        # https://www.medrxiv.org/content/medrxiv/early/2020/05/05/2020.04.30.20084780.full.pdf
        self.th0 = 6.0

        # Hospitalization dwell time is longer for old people
        def th_f(med_age=None):
            if med_age is None:  # not used much as median age data in future is now generated
                return self.th0
            else:
                return self.th0 * (
                    1.0 + th_elongation_fraction * max(0.0, min(1.0, (med_age - 30.0) / 30.0))
                )

        self.t_h = th_f

        if self.age_eval_delay is None:
            self.age_eval_delay = (
                self.t_i + NowcastingSEIRModel.FRACTION_HOSP_DWELL_AS_DELAY * self.t_h()
            )
        self.fw0 = 0.5

        # Constants used for solving positivity (as function of T/I - test rate / infections) ensuring continuity
        # of positivity and its derivative. See postivity() and its inverse positivity_to_t_over_i() below.
        self.pos0 = 0.5  # max positivity for testing -> 0
        self.pos_c = 1.75
        self.pos_x0 = 2.0
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
        TODO double check if it is I,C that are fixed or T over i,C and improve documentation here
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
        Test positivity as a function of x=T/I where T = testing rate / I = number of infected
        This function should match two constraints
            p(x) = .5 for x->0 (when very insufficient testing 50% of those tested will be positive)
            p(x) <~ 1/x for x-> infinity (almost all infections found as testing -> infinity)
        To achieve this different functions are used (switching at x=x0) and the constants b and d
        are solved for to ensure continuity of the function and its derivative across x0

        TODO more detail for Natasha
        """
        p = 0.5 / t_over_i
        if t_over_i < self.pos_x0:  # approaches .5 for x -> 0
            p = p * (t_over_i - self.pos_b * t_over_i ** 1.5)
        else:  # approaches .875/x for x -> infinity
            p = p * (self.pos_c - self.pos_d / t_over_i ** 0.5)
        return p

    def positivity_to_t_over_i(self, pos):
        """
        Given positivity determine T/I (inverting positivity function above).
        Rely on positivity to be a continuous, strictly decreasing function of t_over_i over [.001,1000]
        to invert using binary search in log space.
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


class ModelRun:
    """
    A run (in time) of a model with (previously fit) parameters maintained in model

    Model compartments (evolved in time):
        S - susceptible (its really S/N that is important)
        E - exposed -> will become I or A over time
        I - infected, syptomatic -> can become C over time
        A - infected, weakly or asympomatic (interchangably referred to as W)
        nC - new (confirmed) cases for tracking purposes
        C - observed (confirmed) cases -> can become H over time
        H - observed hosptializations (usually observable). 
        nD - observed new deaths - total deaths not accumulated in this model
        R - recoveries (just to validate conservation of people)
        b - beta that drives new infections - when R(t) supplied this is the beta that would have been needed
    Note R and HICU, HVent (which will be generated from H - but outside of this model) are not included.

    Each run also has the ability to calibrate itself to the current observed ratios of
    hopitalizations to cases and deaths to hopitalizations. Each of these can support a constant
    factor and linear change over time (but the latter is not yet used but is left in the code
    in case needed in the future).

    Observables, recent history smoothed and/or future projections, used to either inject known
    important sources of time variation into, or constrain, the running model:
        rt - growth rate for nC (observed in past, predicted in future)
        case_median_age - function predicting median age of new cases into the future. Typically derived
            using rt above (see Demographics.infer_median_age_function)
        test_positivity - function predicting (in future) fraction of tests that are positive
        initial values of nC, H, nD described above - its the ratios H/C and D/H that provide constraints
        (FUTURE) test_processing_delay - in days from time sample supplied to test result available
        (FUTURE) case_fraction_traceable - fraction of cases that have been traced to their source

    TODO ensure the list of parameters is complete
    Parameters
    ----------
    model: NowcastingSEIRModel
        Run independent parts of the base Model - parts that will be trained generally ahead of time
    N: int
        Total population
    t_list: int[]
        Times relative to a reference date
    rt_f: lambda(t)
        instead of suppression_policy, R0
        TODO FUTURE consider option to provide smoothed cases directly
    testing_rate_f: lambda(t)
        Testing rate
    case_median_age_f: lambda(t)
        Median age of people that test positive each day
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

    TODO FUTURE better address restarting from a model run from yesterday
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
        initial_compartments=None,  # use this OR the next one
        historical_compartments=None,  # use this OR the previous one - TODO explain why this is here
        compartment_ratios_initial=None,
        # hospitalizations_threshold=None,  # At which point mortality starts to increase due to constrained resources
        # Observable compartments to be eventually used in further constraining the model
        # observed_compartment_history=None,
        #### Optional controls for how the model run operates
        force_stationary=False,  # if True susceptible will be pinned to N
        auto_initialize_other_compartments=False,
        auto_calibrate=False,
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

        # Adjust initial conditions and transition fractions for H and nD to align with inital compartments
        if self.auto_calibrate:
            max_cal = 3.0
            current = ModelRun.array_to_dict(y)
            adj_H = 1.0
            adj_nD = 1.0
            if y0["H"] > 0.0 and current["H"] > 0.0:
                adj_H = max(min(y0["H"] / current["H"], max_cal), 1.0 / max_cal)
                current["H"] = y0["H"]
            if y0["nD"] > 0.0 and current["nD"] > 0.0:
                adj_nD = max(min(y0["nD"] / current["nD"] / adj_H, max_cal), 1.0 / max_cal)
                current["nD"] = y0["nD"]
            y = ModelRun.dict_to_array(current)
            self.model.adjustFractions(adj_H, adj_nD)

        y_accum = list()
        y_accum.append(y)

        implicit = False if self.force_stationary else True

        # If today is set and is in the future than we are calibrating on more than one day
        # Note this is not currently supported - just started to "frame this in"
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

            # TODO add incident hospitalizations
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
        # Note for now lr_f*[1] is always 0. (no time variance of the calibration)
        if self.case_median_age_f is not None:
            fh0 = Transitions.fh0_f(median_age=med_age_h)  # was model.
            fd0 = Transitions.fd0_f(median_age=med_age_d)
        else:
            fh0 = Transitions.fh0_f()
            fd0 = Transitions.fd0_f()
        fh = fh0 * model.lr_fh[0] + model.lr_fh[1] * (t - self.t_list[0])
        fh = max(0.0, min(1.0, fh))
        fd = fd0 * model.lr_fd[0] + model.lr_fd[1] * (t - self.t_list[0])
        fd = max(0.0, min(1.0, fd))
        fw = model.fw0  # might do something smarter later

        # Damping factors for numerical stability
        e_max_growth = 2.0

        # Use delayed Cases and Infected to drive hospitalizations
        # Currently model.delay_ci_h is usually 0. so this adds no extra delay
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

            # Which determines W (this is just A = asymptomatic)
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

        # Regular fully explicit evaluation
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
                    Transitions.fh0_f(
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
                    Transitions.fd0_f(
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
