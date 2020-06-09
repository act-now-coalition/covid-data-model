import os
import us
import logging
from pprint import pformat
import datetime as dt
from datetime import datetime, timedelta
from multiprocessing import Pool
from copy import deepcopy

import pandas as pd
import dill as pickle
import numpy as np
import iminuit
from scipy.stats import gamma, norm
from matplotlib import pyplot as plt

from pyseir.models import suppression_policies
from pyseir import load_data
from pyseir.models.seir_model import SEIRModel
from pyseir.models.seir_model_age import SEIRModelAge
from libs.datasets.dataset_utils import AggregationLevel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from pyseir.parameters.parameter_ensemble_generator_age import ParameterEnsembleGeneratorAge
from pyseir.load_data import HospitalizationDataType
from pyseir.utils import get_run_artifact_path, RunArtifact
from pyseir.inference.fit_results import load_inference_result

log = logging.getLogger()


def calc_chi_sq(obs, predicted, stddev):
    return np.sum((obs - predicted) ** 2 / stddev ** 2)


class ModelFitter:
    """
    Fit a SEIR model and suppression policy for a geographic unit (county or
    state) against case, hospitalization, and mortality data.  The error model
    to blend these is dominated by systematic uncertainties rather than
    statistical ones. We allow for relative rates of confirmed cases, hosp and
    deaths to float, allowing the fitter to focus on the shape parameters rather
    than absolutes.

    Parameters
    ----------
    fips: str
        State or county fips code.
    ref_date: datetime
        Date to reference against. This should be before the first case.
    min_deaths: int
        Minimum number of fatalities to use death data.
    n_years: int
        Number of years to run the simulation for.
    cases_to_deaths_err_factor: float
        To control chi2 mix between cases and deaths, we multiply the systematic
        errors of cases by this number. Thus higher numbers reduce the influence
        of case data on the fit.
    hospital_to_deaths_err_factor: float
        To control chi2 mix between hospitalization and deaths, we multiply the
        systematic errors of cases by this number. Thus higher numbers reduce
        the influence of hospitalization data on the fit.
    percent_error_on_max_observation: float
        Relative error on the max observation in each category.  The reltive
        errors are then scaled based on the sqrt(observed value / observed max)
        relative to the max. The overall scale here doesn't influence the max
        likelihood fit, but it does influence the prior blending and error
        estimates.  0.5 = 50% error. Best to be conservatively high.
    with_age_structure: bool
        Whether run model with age structure.
    """

    DEFAULT_FIT_PARAMS = dict(
        R0=3.4,
        limit_R0=[2, 4.5],
        error_R0=0.05,
        log10_I_initial=1,
        limit_log10_I_initial=[0.333, 2],
        error_log10_I_initial=0.33,
        t0=60,
        limit_t0=[10, 80],
        error_t0=2.0,
        eps=0.3,
        limit_eps=[0.20, 1.2],
        error_eps=0.005,
        t_break=20,
        limit_t_break=[5, 40],
        error_t_break=1,
        eps2=0.3,
        limit_eps2=[0.20, 2.0],
        error_eps2=0.005,
        t_delta_phases=14,  # number of days between second and third ramps
        limit_t_delta_phases=[14, 100],  # good as of June 3, 2020 may need to update in the future
        error_t_delta_phases=1,
        test_fraction=0.1,
        limit_test_fraction=[0.02, 1],
        error_test_fraction=0.02,
        hosp_fraction=0.7,
        limit_hosp_fraction=[0.25, 1],
        error_hosp_fraction=0.05,
        # Let's not fit this to start...
        errordef=0.5,
    )

    PARAM_SETS = {
        ("HI", "AK", "MT", "ID", "LA", "ND", "WV", "WY"): dict(
            eps=0.25, t0=75, t_break=10, limit_t0=[50, 90]
        ),
    }

    REFF_LOWER_BOUND = 0.7

    steady_state_exposed_to_infected_ratio = 1.2

    def __init__(
        self,
        fips,
        ref_date=datetime(year=2020, month=1, day=1),
        min_deaths=2,
        n_years=1,
        cases_to_deaths_err_factor=0.5,
        hospital_to_deaths_err_factor=0.5,
        percent_error_on_max_observation=0.5,
        with_age_structure=False,
    ):

        # Seed the random state. It is unclear whether this propagates to the
        # Minuit optimizer.
        np.random.seed(seed=42)

        self.fips = fips
        self.ref_date = ref_date
        self.days_since_ref_date = (dt.date.today() - ref_date.date() - timedelta(days=7)).days
        # ndays end of 2nd ramp may extend past days_since_ref_date w/o  penalty on chi2 score
        self.days_allowed_beyond_ref = 0
        self.min_deaths = min_deaths
        self.t_list = np.linspace(0, int(365 * n_years), int(365 * n_years) + 1)
        self.cases_to_deaths_err_factor = cases_to_deaths_err_factor
        self.hospital_to_deaths_err_factor = hospital_to_deaths_err_factor
        self.percent_error_on_max_observation = percent_error_on_max_observation
        self.t0_guess = 60
        self.with_age_structure = with_age_structure

        if len(fips) == 2:  # State FIPS are 2 digits
            self.agg_level = AggregationLevel.STATE
            self.state_obj = us.states.lookup(self.fips)
            self.state = self.state_obj.name

            (
                self.times,
                self.observed_new_cases,
                self.observed_new_deaths,
            ) = load_data.load_new_case_data_by_state(self.state, self.ref_date)

            (
                self.hospital_times,
                self.hospitalizations,
                self.hospitalization_data_type,
            ) = load_data.load_hospitalization_data_by_state(self.state_obj.abbr, t0=self.ref_date)
            self.display_name = self.state
        else:
            self.agg_level = AggregationLevel.COUNTY
            geo_metadata = load_data.load_county_metadata().set_index("fips").loc[fips].to_dict()
            state = geo_metadata["state"]
            self.state_obj = us.states.lookup(state)
            county = geo_metadata["county"]
            if county:
                self.display_name = county + ", " + state
            else:
                self.display_name = state
            # TODO Swap for new data source.
            (
                self.times,
                self.observed_new_cases,
                self.observed_new_deaths,
            ) = load_data.load_new_case_data_by_fips(self.fips, t0=self.ref_date)
            (
                self.hospital_times,
                self.hospitalizations,
                self.hospitalization_data_type,
            ) = load_data.load_hospitalization_data(self.fips, t0=self.ref_date)

        self.cases_stdev, self.hosp_stdev, self.deaths_stdev = self.calculate_observation_errors()
        self.set_inference_parameters()

        self.model_fit_keys = [
            "R0",
            "eps",
            "t_break",
            "eps2",
            "t_delta_phases",
            "log10_I_initial",
        ]

        self.SEIR_kwargs = self.get_average_seir_parameters()
        self.fit_results = None
        self.mle_model = None

        self.chi2_deaths = None
        self.chi2_cases = None
        self.chi2_hosp = None
        self.dof_deaths = None
        self.dof_cases = None
        self.dof_hosp = None

    def set_inference_parameters(self):
        """
        Setup inference parameters based on data availability and manual
        overrides.  As data becomes more sparse, we further constrain the fit,
        which improves stability substantially.
        """
        self.fit_params = self.DEFAULT_FIT_PARAMS
        # Update any state specific params.
        for k, v in self.PARAM_SETS.items():
            if self.state_obj.abbr in k:
                self.fit_params.update(v)

        self.fit_params["fix_hosp_fraction"] = self.hospitalizations is None
        if self.fit_params["fix_hosp_fraction"]:
            self.fit_params["hosp_fraction"] = 1

        if len(self.fips) == 5:
            OBSERVED_NEW_CASES_GUESS_THRESHOLD = 2
            idx_enough_cases = np.argwhere(
                np.cumsum(self.observed_new_cases) >= OBSERVED_NEW_CASES_GUESS_THRESHOLD
            )[0][0]
            initial_cases_guess = np.cumsum(self.observed_new_cases)[idx_enough_cases]
            t0_guess = list(self.times)[idx_enough_cases]

            state_fit_result = load_inference_result(fips=self.state_obj.fips)
            self.fit_params["t0"] = t0_guess

            total_cases = np.sum(self.observed_new_cases)
            self.fit_params["log10_I_initial"] = np.log10(
                initial_cases_guess / self.fit_params["test_fraction"]
            )
            self.fit_params["limit_t0"] = state_fit_result["t0"] - 20, state_fit_result["t0"] + 30
            self.fit_params["t_break"] = state_fit_result["t_break"] - (
                t0_guess - state_fit_result["t0"]
            )
            self.fit_params["R0"] = state_fit_result["R0"]
            self.fit_params["test_fraction"] = state_fit_result["test_fraction"]
            self.fit_params["eps"] = state_fit_result["eps"]
            if total_cases < 100:
                self.fit_params["t_break"] = 10
                self.fit_params["fix_test_fraction"] = True
                self.fit_params["fix_R0"] = True
                self.fit_params["limit_t0"] = (
                    state_fit_result["t0"] - 5,
                    state_fit_result["t0"] + 30,
                )
            if total_cases < 50:
                self.fit_params["fix_eps"] = True
                self.fit_params["fix_t_break"] = True

    def get_average_seir_parameters(self):
        """
        Generate the additional fitter candidates from the ensemble generator. This
        has the suppression policy and R0 keys removed.

        Returns
        -------
        SEIR_kwargs: dict
            The average ensemble params.
        """
        if self.with_age_structure:
            parameter_generator = ParameterEnsembleGeneratorAge
        else:
            parameter_generator = ParameterEnsembleGenerator

        SEIR_kwargs = parameter_generator(
            fips=self.fips, N_samples=5000, t_list=self.t_list, suppression_policy=None
        ).get_average_seir_parameters()

        SEIR_kwargs = {k: v for k, v in SEIR_kwargs.items() if k not in self.fit_params}
        del SEIR_kwargs["suppression_policy"]
        del SEIR_kwargs["I_initial"]
        return SEIR_kwargs

    def calculate_observation_errors(self):
        """
        Generate the errors on the observations.
        Here we throw out a few assumptions to plant a flag...
        1. Systematic errors dominate and are likely of order 50% at least based
        on 100% undercounting of deaths and hospitalizations in many places.
        Though we account for static undercounting by letting case and hosp
        counts float, so lets assume the error is a bit smaller for now.
        2. 100% is too small for 1 case count or mortality.. We should be much
           more confident in large numbers of observations
        3. TODO: Deal with this fact.. Actual observations are lower bounds.
                 Need asymmetric errors.
        As an error model, absolutes are less important to our problem compared
        to getting relative error scaling reasonably done. This is not true if
        drawing contours and confidence intervals which is why we choose large
        conservative errors to overestimate the uncertainty.
        As a relative scaling model we think about Poisson processes and scale
        the errors in the following way:
        1. Set the error of the largest observation to 100% of its value.
        2. Scale all other errors based on sqrt(value) * sqrt(max_value)

        Returns
        -------
        cases_stdev: array-like
            Float uncertainties (stdev) for case data.
        hosp_stdev: array-like
            Float uncertainties (stdev) for hosp data.
        deaths_stdev: array-like
            Float uncertainties (stdev) for death data.
            Float uncertainties (stdev) for death data.
        """
        # Stdev 50% of values.
        cases_stdev = (
            self.percent_error_on_max_observation
            * self.cases_to_deaths_err_factor
            * self.observed_new_cases ** 0.5
            * self.observed_new_cases.max() ** 0.5
        )
        deaths_stdev = (
            self.percent_error_on_max_observation
            * self.observed_new_deaths ** 0.5
            * self.observed_new_deaths.max() ** 0.5
        )

        # Add a bit more error in cases with very few deaths, cases, or hosps. Specifically, we
        # inflate error bars for very small numbers of deaths, cases, and hosps
        # since these clearly reduce the fit accuracy (and individual events are
        # rife with systematic issues).
        deaths_stdev[self.observed_new_deaths <= 4] = (
            deaths_stdev[self.observed_new_deaths <= 4] * 3
        )
        cases_stdev[self.observed_new_cases <= 4] = cases_stdev[self.observed_new_cases <= 4] * 3

        # If cumulative hospitalizations, differentiate.
        if self.hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
            hosp_data = (self.hospitalizations[1:] - self.hospitalizations[:-1]).clip(min=0)
            hosp_stdev = (
                self.percent_error_on_max_observation
                * self.hospital_to_deaths_err_factor
                * hosp_data ** 0.5
                * hosp_data.max() ** 0.5
            )
            # Increase errors a bit for very low hospitalizations.
            # There are clear outliers due to data quality.
            hosp_stdev[hosp_data <= 2] *= 3

        elif self.hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
            hosp_data = self.hospitalizations
            hosp_stdev = (
                self.percent_error_on_max_observation
                * self.hospital_to_deaths_err_factor
                * hosp_data ** 0.5
                * hosp_data.max() ** 0.5
            )
            # Increase errors a bit for very low hospitalizations.
            # There are clear outliers due to data quality.
            hosp_stdev[hosp_data <= 2] *= 3
        else:
            hosp_stdev = None

        # Zero inflated poisson. This is set to a value that still provides some
        # constraints toward zero.
        cases_stdev[cases_stdev == 0] = 1e2
        deaths_stdev[deaths_stdev == 0] = 1e2
        if hosp_stdev is not None:
            hosp_stdev[hosp_stdev == 0] = 1e2

        return cases_stdev, hosp_stdev, deaths_stdev

    def run_model(self, R0, eps, t_break, eps2, t_delta_phases, log10_I_initial):
        """
        Generate the model and run.

        Parameters
        ----------
        R0: float
            Basic reproduction number
        eps: float
            Fraction of reduction in contact rates in the second stage.
        t_break: float
            Timing for the switch in suppression policy.
        eps2: float
            Fraction of reduction in contact rates in the third stage
        t_delta_phases: float
            Timing for the switch in from second to third stage.
        log10_I_initial:
            log10 initial infections.

        Returns
        -------
        model: SEIRModel
            The SEIR model that has been run.
        """

        suppression_policy = suppression_policies.get_epsilon_interpolator(
            eps, t_break, eps2, t_delta_phases
        )

        if self.with_age_structure:
            age_distribution = self.SEIR_kwargs["N"] / self.SEIR_kwargs["N"].sum()
            seir_model = SEIRModelAge
        else:
            age_distribution = 1
            seir_model = SEIRModel

        # Load up some number of initial exposed so the initial flow into
        # infected is stable.
        self.SEIR_kwargs["E_initial"] = (
            self.steady_state_exposed_to_infected_ratio * 10 ** log10_I_initial * age_distribution
        )

        model = seir_model(
            R0=R0,
            suppression_policy=suppression_policy,
            I_initial=10 ** log10_I_initial * age_distribution,
            **self.SEIR_kwargs,
        )

        model.run()
        return model

    def _fit_seir(
        self,
        R0,
        t0,
        eps,
        t_break,
        eps2,
        t_delta_phases,
        test_fraction,
        hosp_fraction,
        log10_I_initial,
    ):
        """
        Fit SEIR model by MLE.

        Parameters
        ----------
        R0: float
            Basic reproduction number
        t0: float
            Epidemic starting time.
        eps: float
            Fraction of reduction in contact rates as result of  to suppression
            policy projected into future.
        t_break: float
            Timing for the switch in suppression policy.
        test_fraction: float
            Fraction of cases that get tested.
        hosp_fraction: float
            Fraction of actual hospitalizations vs the total.
        log10_I_initial:
            log10 initial infections.

        Returns
        -------
          : float
            Chi square of fitting model to observed cases, deaths, and hospitalizations.
        """
        l = locals()
        model_kwargs = {k: l[k] for k in self.model_fit_keys}

        # Last data point in ramp 2
        last_data_point_ramp_2 = t0 + t_break + 14 + t_delta_phases + 14
        # Number of future days used in second ramp period
        number_of_not_allowed_days_used = last_data_point_ramp_2 - self.days_since_ref_date
        # Multiplicative chi2 penalty if future_days are used in second ramp period (set to 1 by default)
        not_allowed_days_penalty = 0.0

        # If using more future days than allowed, updated not_allowed_days_penalty
        if number_of_not_allowed_days_used > self.days_allowed_beyond_ref:
            not_allowed_days_penalty = 10 * number_of_not_allowed_days_used

        model = self.run_model(**model_kwargs)
        # -----------------------------------
        # Chi2 Cases
        # -----------------------------------
        # Extract the predicted rates from the model.
        predicted_cases = (
            test_fraction
            * model.gamma
            * np.interp(
                self.times, self.t_list + t0, model.results["total_new_infections"], left=0, right=0
            )
        )
        chi2_cases = calc_chi_sq(self.observed_new_cases, predicted_cases, self.cases_stdev)

        # -----------------------------------
        # Chi2 Hospitalizations
        # -----------------------------------
        if self.hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
            predicted_hosp = hosp_fraction * np.interp(
                self.hospital_times,
                self.t_list + t0,
                model.results["HGen"] + model.results["HICU"],
                left=0,
                right=0,
            )
            chi2_hosp = calc_chi_sq(self.hospitalizations, predicted_hosp, self.hosp_stdev)
            self.dof_hosp = (self.observed_new_cases > 0).sum()

        elif self.hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
            # Cumulative, so differentiate the data
            cumulative_hosp_predicted = (
                model.results["HGen_cumulative"] + model.results["HICU_cumulative"]
            )
            new_hosp_predicted = cumulative_hosp_predicted[1:] - cumulative_hosp_predicted[:-1]
            new_hosp_predicted = hosp_fraction * np.interp(
                self.hospital_times[1:], self.t_list[1:] + t0, new_hosp_predicted, left=0, right=0
            )
            new_hosp_observed = self.hospitalizations[1:] - self.hospitalizations[:-1]

            chi2_hosp = calc_chi_sq(new_hosp_observed, new_hosp_predicted, self.hosp_stdev)
            self.dof_hosp = (self.observed_new_cases > 0).sum()
        else:
            chi2_hosp = 0
            self.dof_hosp = 1e-10

        # -----------------------------------
        # Chi2 Deaths
        # -----------------------------------
        # Only use deaths if there are enough observations..
        predicted_deaths = np.interp(
            self.times, self.t_list + t0, model.results["total_deaths_per_day"], left=0, right=0
        )
        if self.observed_new_deaths.sum() > self.min_deaths:
            chi2_deaths = calc_chi_sq(self.observed_new_deaths, predicted_deaths, self.deaths_stdev)
        else:
            chi2_deaths = 0

        self.chi2_deaths = chi2_deaths
        self.chi2_cases = chi2_cases
        self.chi2_hosp = chi2_hosp
        self.dof_deaths = (self.observed_new_deaths > 0).sum()
        self.dof_cases = (self.observed_new_cases > 0).sum()

        not_penalized_score = chi2_deaths + chi2_cases + chi2_hosp

        # Calculate the final score as the product of the not_allowed_days_penalty and not_penalized_score
        score = not_allowed_days_penalty + (chi2_deaths + chi2_cases + chi2_hosp)

        return score

    @staticmethod
    def get_posterior_estimate_eps(R0, eps, eps_error, lower_bound_reff, plot=False):
        """
        Generate a posterior estimate for epsilon based on the inferred R0. This
        is a little weird right now since we actually want a prior on Reff. So
        in this case we use the inferred R0 to convert eps -> Reff, apply a
        prior, and invert this transform to get back to the epsilon Max
        A-Posteriori (MAP) estimate.

        Parameters
        ----------
        R0:
            Stage one reproductive rate
        eps:
            Current stage's relative ratio to R0
        eps_error:
            Error in eps estimate (currently from MIGRAD)
        lower_bound_reff:
            The lower bound on reff to be returned.

        Returns
        -------
        posterior_map_estimate: float
            Max A-Posteriori (MAP) estimate for epsilon.
        """
        EPS_ERROR_FLOOR = 0.05  # Sometimes this is estimated to be way to small (incorrectly since
        # we don't know the true error model). This is a problem for bayesian updates. Set a lower
        # bound for the error here.

        R_eff = R0 * eps
        R_eff_stdev = R0 * max(eps_error, EPS_ERROR_FLOOR)

        x = np.linspace(0.00, 10, 1001)
        delta_x = x[1] - x[0]

        # TODO: Extract and Label Gamma Scaling Factors So Others Can Understand
        prior = gamma.pdf((x - lower_bound_reff) / 1.5, 1.1)
        # Add a tiny amount to the likelihood to prevent zero common support
        # between the prior and likelihood functions.
        likelihood = norm.pdf(x, R_eff, R_eff_stdev) + 0.0001
        posterior = prior * likelihood
        posterior = posterior / (posterior.sum() * delta_x)
        posterior_MAP_estimate = x[np.argmax(posterior)] / R0

        if plot:
            plt.plot(x, prior, label="Prior")
            plt.plot(x, likelihood, label="Likelihood")
            plt.plot(x, posterior, label="Posterior")
            plt.grid()
            plt.legend()

        return posterior_MAP_estimate

    def fit(self):
        """
        Fit a model to the data.
        """
        minuit = iminuit.Minuit(self._fit_seir, **self.fit_params, print_level=1)

        if os.environ.get("PYSEIR_FAST_AND_DIRTY"):
            minuit.strategy = 0

        # run MIGRAD algorithm for optimization.
        # for details refer: https://root.cern/root/html528/TMinuit.html
        minuit.migrad(precision=1e-6)
        self.fit_results = dict(fips=self.fips, **dict(minuit.values))
        self.fit_results.update({k + "_error": v for k, v in dict(minuit.errors).items()})

        # This just updates chi2 values
        self._fit_seir(**dict(minuit.values))

        if self.fit_results["eps"] < 0.1:
            raise RuntimeError(
                f"Fit failed for {self.state, self.fips}: "
                f"Epsilon == 0 which implies lack of convergence."
            )

        # Most naive constraints: apply the same constraint to both epsilon 2 and epsilon 3
        for epsilon in ["eps", "eps2"]:
            adjusted_epsilon = ModelFitter.get_posterior_estimate_eps(
                R0=self.fit_results["R0"],
                eps=self.fit_results[epsilon],
                eps_error=self.fit_results[f"{epsilon}_error"],
                lower_bound_reff=ModelFitter.REFF_LOWER_BOUND,
            )
            # TODO: Add structured logging if this change is significant
            self.fit_results[epsilon] = adjusted_epsilon

        if np.isnan(self.fit_results["t0"]):
            logging.error(f"Could not compute MLE values for {self.display_name}")
            self.fit_results["t0_date"] = (
                self.ref_date + timedelta(days=self.t0_guess)
            ).isoformat()
        else:
            self.fit_results["t0_date"] = (
                self.ref_date + timedelta(days=self.fit_results["t0"])
            ).isoformat()
        self.fit_results["t_today"] = (datetime.today() - self.ref_date).days

        self.fit_results["Reff"] = self.fit_results["R0"] * self.fit_results["eps"]
        self.fit_results["Reff2"] = self.fit_results["R0"] * self.fit_results["eps2"]

        self.fit_results["chi2_cases"] = self.chi2_cases
        if self.hospitalizations is not None:
            self.fit_results["chi2_hosps"] = self.chi2_hosp
        self.fit_results["chi2_deaths"] = self.chi2_deaths

        if self.hospitalization_data_type:
            self.fit_results["hospitalization_data_type"] = self.hospitalization_data_type.value
        else:
            self.fit_results["hospitalization_data_type"] = self.hospitalization_data_type

        try:
            param_state = minuit.get_param_states()
            logging.info(f"Fit Results for {self.display_name} \n {param_state}")
        except:
            param_state = dict(minuit.values)
            logging.info(f"Fit Results for {self.display_name} \n {param_state}")

        logging.info(f"Complete fit results for {self.display_name} \n {pformat(self.fit_results)}")
        self.mle_model = self.run_model(**{k: self.fit_results[k] for k in self.model_fit_keys})

    def plot_fitting_results(self):
        """
        Plotting model fitting results.
        """
        data_dates = [self.ref_date + timedelta(days=t) for t in self.times]
        if self.hospital_times is not None:
            hosp_dates = [self.ref_date + timedelta(days=float(t)) for t in self.hospital_times]
        model_dates = [
            self.ref_date + timedelta(days=t + self.fit_results["t0"]) for t in self.t_list
        ]

        # Don't display the zero-inflated error bars
        cases_err = np.array(self.cases_stdev)
        cases_err[self.observed_new_cases == 0] = 0
        death_err = deepcopy(self.deaths_stdev)
        death_err[self.observed_new_deaths == 0] = 0
        if self.hosp_stdev is not None:
            hosp_stdev = deepcopy(self.hosp_stdev)
            hosp_stdev[hosp_stdev > 1e5] = 0

        plt.figure(figsize=(18, 12))
        plt.errorbar(
            data_dates,
            self.observed_new_cases,
            yerr=cases_err,
            marker="o",
            linestyle="",
            label="Observed Cases Per Day",
            color="steelblue",
            capsize=3,
            alpha=0.4,
            markersize=10,
        )
        plt.errorbar(
            data_dates,
            self.observed_new_deaths,
            yerr=death_err,
            marker="d",
            linestyle="",
            label="Observed Deaths Per Day",
            color="firebrick",
            capsize=3,
            alpha=0.4,
            markersize=10,
        )

        plt.plot(
            model_dates,
            self.mle_model.results["total_new_infections"],
            label="Estimated Total New Infections Per Day",
            linestyle="--",
            lw=4,
            color="steelblue",
        )
        plt.plot(
            model_dates,
            self.fit_results["test_fraction"] * self.mle_model.results["total_new_infections"],
            label="Estimated Tested New Infections Per Day",
            color="steelblue",
            lw=4,
        )

        plt.plot(
            model_dates,
            self.mle_model.results["total_deaths_per_day"],
            label="Model Deaths Per Day",
            color="firebrick",
            lw=4,
        )

        if self.hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
            new_hosp_observed = self.hospitalizations[1:] - self.hospitalizations[:-1]
            plt.errorbar(
                hosp_dates[1:],
                new_hosp_observed,
                yerr=hosp_stdev,
                marker="s",
                linestyle="",
                label="Observed New Hospitalizations Per Day",
                color="darkseagreen",
                capsize=3,
                alpha=1,
            )
            predicted_hosp = (
                self.mle_model.results["HGen_cumulative"]
                + self.mle_model.results["HICU_cumulative"]
            )
            predicted_hosp = predicted_hosp[1:] - predicted_hosp[:-1]
            plt.plot(
                model_dates[1:],
                self.fit_results["hosp_fraction"] * predicted_hosp,
                label="Estimated Total New Hospitalizations Per Day",
                linestyle="-.",
                lw=4,
                color="darkseagreen",
                markersize=10,
            )
        elif self.hospitalization_data_type is HospitalizationDataType.CURRENT_HOSPITALIZATIONS:
            plt.errorbar(
                hosp_dates,
                self.hospitalizations,
                yerr=hosp_stdev,
                marker="s",
                linestyle="",
                label="Observed Total Current Hospitalizations",
                color="darkseagreen",
                capsize=3,
                alpha=0.5,
                markersize=10,
            )
            predicted_hosp = self.mle_model.results["HGen"] + self.mle_model.results["HICU"]
            plt.plot(
                model_dates,
                self.fit_results["hosp_fraction"] * predicted_hosp,
                label="Estimated Total Current Hospitalizations",
                linestyle="-.",
                lw=4,
                color="darkseagreen",
            )

        plt.plot(
            model_dates,
            self.fit_results["hosp_fraction"] * self.mle_model.results["HICU"],
            label="Estimated ICU Occupancy",
            linestyle=":",
            lw=6,
            color="black",
        )
        plt.plot(
            model_dates,
            self.fit_results["hosp_fraction"] * self.mle_model.results["HGen"],
            label="Estimated General Occupancy",
            linestyle=":",
            lw=4,
            color="black",
            alpha=0.4,
        )

        plt.yscale("log")
        y_lim = plt.ylim(0.8e0)

        start_intervention_date = self.ref_date + timedelta(
            days=self.fit_results["t_break"] + self.fit_results["t0"]
        )
        stop_intervention_date = start_intervention_date + timedelta(days=14)

        plt.fill_betweenx(
            [y_lim[0], y_lim[1]],
            [start_intervention_date, start_intervention_date],
            [stop_intervention_date, stop_intervention_date],
            alpha=0.2,
            label="Estimated Intervention",
        )

        start_intervention2_date = (
            self.ref_date
            + timedelta(
                days=self.fit_results["t_break"]
                + self.fit_results["t_delta_phases"]
                + self.fit_results["t0"]
            )
            + timedelta(days=14)
        )
        stop_intervention2_date = start_intervention2_date + timedelta(days=14)

        plt.fill_betweenx(
            [y_lim[0], y_lim[1]],
            [start_intervention2_date, start_intervention2_date],
            [stop_intervention2_date, stop_intervention2_date],
            alpha=0.2,
            label="Estimated Intervention2",
        )

        running_total = timedelta(days=0)
        for i_label, k in enumerate(
            (
                "symptoms_to_hospital_days",
                "hospitalization_length_of_stay_general",
                "hospitalization_length_of_stay_icu",
            )
        ):

            end_time = timedelta(days=self.SEIR_kwargs[k])
            x = start_intervention_date + running_total
            y = 1.5 ** (i_label + 1)
            plt.errorbar(
                x=[x],
                y=[y],
                xerr=[[timedelta(days=0)], [end_time]],
                marker="",
                capsize=8,
                color="k",
                elinewidth=3,
                capthick=3,
            )
            plt.text(
                x + (end_time + timedelta(days=2)), y, k.replace("_", " ").title(), fontsize=14
            )
            running_total += end_time

        if self.SEIR_kwargs["beds_ICU"] > 0:
            plt.hlines(
                self.SEIR_kwargs["beds_ICU"],
                *plt.xlim(),
                color="k",
                linestyles="-",
                linewidths=6,
                alpha=0.2,
            )
            plt.text(
                data_dates[0] + timedelta(days=5),
                self.SEIR_kwargs["beds_ICU"] * 1.1,
                "Available ICU Capacity",
                color="k",
                alpha=0.5,
                fontsize=15,
            )

        plt.ylim(*y_lim)
        plt.xlim(min(model_dates[0], data_dates[0]), data_dates[-1] + timedelta(days=150))
        plt.xticks(rotation=30, fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc=4, fontsize=14)
        plt.grid(which="both", alpha=0.5)
        plt.title(self.display_name, fontsize=60)

        chi_total = 0
        for i, (k, v) in enumerate(self.fit_results.items()):
            if k in ("chi2_cases", "chi2_deaths", "chi2_hosps"):
                chi_total += v

        for i, (k, v) in enumerate(self.fit_results.items()):

            fontweight = (
                "bold"
                if k in ("R0", "Reff", "Reff2", "eps", "eps2", "t_delta_phases")
                else "normal"
            )

            if np.isscalar(v) and not isinstance(v, str):
                plt.text(
                    1.05,
                    0.7 - 0.032 * i,
                    f"{k}={v:1.3f}",
                    transform=plt.gca().transAxes,
                    fontsize=15,
                    alpha=0.6,
                    fontweight=fontweight,
                )

            else:
                plt.text(
                    1.05,
                    0.7 - 0.032 * i,
                    f"{k}={v}",
                    transform=plt.gca().transAxes,
                    fontsize=15,
                    alpha=0.6,
                    fontweight=fontweight,
                )
        plt.text(
            1.05,
            0.75,
            f"total_chi2:{chi_total:1.3f}",
            transform=plt.gca().transAxes,
            fontsize=15,
            alpha=0.6,
            fontweight="bold",
        )
        output_file = get_run_artifact_path(self.fips, RunArtifact.MLE_FIT_REPORT)
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

        self.mle_model.plot_results()
        plt.savefig(output_file.replace("mle_fit_results", "mle_fit_model"), bbox_inches="tight")
        plt.close()

    @classmethod
    def run_for_fips(cls, fips, n_retries=3, with_age_structure=False):
        """
        Run the model fitter for a state or county fips code.

        Parameters
        ----------
        fips: str
            2-digit state or 5-digit county fips code.
        n_retries: int
            The model fitter is stochastic in nature and a seed cannot be set.
            This is a bandaid until more sophisticated retries can be
            implemented.
        with_age_structure: bool
            If True run model with age structure.

        Returns
        -------
        : ModelFitter
        """
        # Assert that there are some cases for counties
        if len(fips) == 5:
            _, observed_new_cases, _ = load_data.load_new_case_data_by_fips(
                fips, t0=datetime.today()
            )
            if observed_new_cases.sum() < 1:
                return None

        try:
            retries_left = n_retries
            model_is_empty = True
            while retries_left > 0 and model_is_empty:
                model_fitter = cls(fips=fips, with_age_structure=with_age_structure)
                try:
                    model_fitter.fit()
                    if model_fitter.mle_model and os.environ.get("PYSEIR_PLOT_RESULTS") == "True":
                        model_fitter.plot_fitting_results()
                except RuntimeError as e:
                    logging.warning("No convergence.. Retrying " + str(e))
                retries_left = retries_left - 1
                if model_fitter.mle_model:
                    model_is_empty = False
            if retries_left <= 0 and model_is_empty:
                raise RuntimeError(f"Could not converge after {n_retries} for fips {fips}")
            return model_fitter
        except Exception:
            logging.exception(f"Failed to run {fips}")
            return None


def _execute_model_for_fips(fips):
    if fips:
        model_fitter = ModelFitter.run_for_fips(fips)
        return model_fitter
    logging.warning(f"Not running model run for ${fips}")
    return None


def _persist_results_per_state(state_df):
    county_output_file = get_run_artifact_path(state_df.fips[0], RunArtifact.MLE_FIT_RESULT)
    data = state_df.drop(["state", "mle_model"], axis=1)
    data.to_json(county_output_file)

    for fips, county_series in state_df.iterrows():
        with open(get_run_artifact_path(fips, RunArtifact.MLE_FIT_MODEL), "wb") as f:
            pickle.dump(county_series.mle_model, f)


def build_county_list(state):
    """
    Build the and return the fips list
    """
    state_obj = us.states.lookup(state)
    logging.info(f"Get fips list for state {state_obj.name}")

    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist["inference_ok"] == True]

    all_fips = df_whitelist[
        df_whitelist["state"].str.lower() == state_obj.name.lower()
    ].fips.tolist()

    return all_fips


def run_state(state, states_only=False, with_age_structure=False):
    """
    Run the fitter for each county in a state.

    Parameters
    ----------
    state: str
        State to run against.
    states_only: bool
        If True only run the state level.
    with_age_structure: bool
        If True run model with age structure.
    """
    state_obj = us.states.lookup(state)
    logging.info(f"Running MLE fitter for state {state_obj.name}")

    model_fitter = ModelFitter.run_for_fips(
        fips=state_obj.fips, with_age_structure=with_age_structure
    )

    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist["inference_ok"] == True]

    output_path = get_run_artifact_path(state_obj.fips, RunArtifact.MLE_FIT_RESULT)
    data = pd.DataFrame(model_fitter.fit_results, index=[state_obj.fips])
    data.to_json(output_path)

    with open(get_run_artifact_path(state_obj.fips, RunArtifact.MLE_FIT_MODEL), "wb") as f:
        pickle.dump(model_fitter.mle_model, f)

    # Run the counties.
    if not states_only:
        # TODO: Replace with build_county_list
        df_whitelist = load_data.load_whitelist()
        df_whitelist = df_whitelist[df_whitelist["inference_ok"] == True]

        all_fips = df_whitelist[
            df_whitelist["state"].str.lower() == state_obj.name.lower()
        ].fips.values

        if len(all_fips) > 0:
            p = Pool()
            fitters = p.map(ModelFitter.run_for_fips, all_fips)
            p.close()

            county_output_file = get_run_artifact_path(all_fips[0], RunArtifact.MLE_FIT_RESULT)
            data = pd.DataFrame([fit.fit_results for fit in fitters if fit])
            data.to_json(county_output_file)

            # Serialize the model results.
            for fips, fitter in zip(all_fips, fitters):
                if fitter:
                    with open(get_run_artifact_path(fips, RunArtifact.MLE_FIT_MODEL), "wb") as f:
                        pickle.dump(fitter.mle_model, f)
