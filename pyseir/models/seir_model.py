import numpy as np

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

        # This compartment is for tracking ventillator count. The beds are
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

    def plot_results(self, y_scale="log", xlim=None):
        """
        Generate a summary plot for the simulation.

        Parameters
        ----------
        y_scale: str
            Matplotlib scale to use on y-axis. Typically 'log' or 'linear'
        """
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        fig = plt.figure(facecolor="w", figsize=(20, 6))
        plt.subplot(131)
        plt.plot(self.t_list, self.results["S"], alpha=1, lw=2, label="Susceptible")
        plt.plot(self.t_list, self.results["E"], alpha=0.5, lw=2, label="Exposed")
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

        # Total for ensuring all compartments sum to 1.
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

        plt.xlabel("Time [days]", fontsize=12)
        plt.yscale(y_scale)
        plt.grid(True, which="both", alpha=0.35)
        plt.legend(framealpha=0.5)
        if xlim:
            plt.xlim(*xlim)
        else:
            plt.xlim(0, self.t_list.max())
        plt.ylim(1, self.N * 1.1)

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

        # Reproduction numbers
        plt.subplot(133)
        plt.plot(self.t_list, [self.suppression_policy(t) for t in self.t_list], c="steelblue")
        plt.ylabel("Contact Rate Reduction")
        plt.xlabel("Time [days]", fontsize=12)
        plt.grid(True, which="both")
        return fig
