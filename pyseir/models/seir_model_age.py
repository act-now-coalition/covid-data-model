import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class SEIRModelAge:
    """
    This class implements a SEIR-like compartmental epidemic model
    consisting of SEIR states plus death, and hospitalizations and age
    structure.

    In the diff eq modeling, these parameters are assumed exponentially
    distributed and modeling occurs in the thermodynamic limit,
    i.e. we do not perform monte carlo for individual cases.

    Parameters
    ----------
    N: np.array
        Total population per age group.
    t_list: array-like
        Array of timesteps. Usually these are spaced daily.
    suppression_policy: callable
        Suppression_policy(t) should return a scalar in [0, 1] which
        represents the contact rate reduction from social distancing.
    A_initial: np.array
        Initial asymptomatic per age group
    I_initial: np.array
        Initial infections per age group
    R_initial: int
        Initial recovered (combining all age groups)
    E_initial: int
        Initial exposed per age group
    HGen_initial: int
        Initial number of General hospital admissions per age group
    HICU_initial: int
        Initial number of ICU cases per age group
    HICUVent_initial: int
        Initial number of ICU cases per age group
    D_initial: int
        Initial number of deaths (combining all age groups)
    n_days: int
        Number of days to simulate.
    birth_rate : float
        Birth per capita per day
    natural_death_rate : float
        Fatility rate due to natural cause.
    max_age: int
        Age upper limit.
    age_steps : np.array
        Time people spend in each age group.
        Last age bin edge is assumed to be 120 years old.
    age_groups : np.array
        Age groups, e.g.
    num_compartments_by_age: int
        Number of compartments with age structure.
        Default 7: S, E, A, I, HGen, HICU, HICUVent.
    num_compartments_not_by_age: int
        Number of compartments without age structure.
        Default 7: R, D, D_no_hgen, D_no_icu, HAdmissions_general,
        HAdmissions_ICU, TotalAllInfections.
    R0: float
        Basic Reproduction number
    R0_hospital: float
        Basic Reproduction number in the hospital.
    kappa: float
        Fractional contact rate for those with symptoms since they
        should be
        isolated vs asymptomatic who are less isolated. A value 1
        implies
        the same rate. A value 0 implies symptomatic people never infect
        others.
    sigma: float
        Latent decay scale is defined as 1 / incubation period.
        1 / 4.8: https://www.imperial.ac.uk/media/imperial-college
        /medicine/sph/ide/gida-fellowships/Imperial-College-COVID19
        -Global-Impact-26-03-2020.pdf
        1 / 5.2 [3, 8]: https://arxiv.org/pdf/2003.10047.pdf
    delta : float
        1 / infectious period.
    delta_hospital: float
        Infectious period for patients in the hospital which is
        usually a bit
        longer.
    gamma: float
        Clinical outbreak rate (fraction of infected that show symptoms)
    contact_matrix : np.array
        With cell at ith row and jth column as contact rate made by ith
        age group with jth age group.
    approximate_R0: bool
        If True calculate R(t) as funciton of initial R0 and suppression policy.
    hospitalization_rate_general: np.array
        Fraction of infected that are hospitalized generally (not in
        ICU)
        by age group.
    hospitalization_rate_icu: np.array
        Fraction of infected that are hospitalized in the ICU
        by age group.
    hospitalization_length_of_stay_icu_and_ventilator: float
        Mean LOS for those requiring ventilators
    fraction_icu_requiring_ventilator: float
        Of the ICU cases, which require ventilators.
    beds_general: int
        General (non-ICU) hospital beds available.
    beds_ICU: int
        ICU beds available
    ventilators: int
        Ventilators available.
    symptoms_to_hospital_days: float
        Mean number of days elapsing between infection and
        hospital admission.
    symptoms_to_mortality_days: float
        Mean number of days for an infected individual to die.
        Hospitalization to death Needs to be added to time to
        15.16 [0, 42] - https://arxiv.org/pdf/2003.10047.pdf
    hospitalization_length_of_stay_general: float
        Mean number of days for a hospitalized individual to be
        discharged.
    hospitalization_length_of_stay_icu
        Mean number of days for a ICU hospitalized individual to be
        discharged.
    mortality_rate_no_ICU_beds: float
        The percentage of those requiring ICU that die if ICU beds
        are not available.
    mortality_rate_no_ventilator: float
        The percentage of those requiring ventilators that die if
        they are not available.
    mortality_rate_no_general_beds: float
        The percentage of those requiring general hospital beds that
        die if they are not available.
    mortality_rate_from_ICU: np.array
        Mortality rate among patients admitted to ICU by age group.
    mortality_rate_from_ICUVent: float
        Mortality rate among patients admitted to ICU with ventilator.
    initial_hospital_bed_utilization: float
        Starting utilization fraction for hospital beds and ICU beds.
    """

    def __init__(self,
                 N,
                 t_list,
                 suppression_policy,
                 A_initial=np.array([1] * 18),
                 I_initial=np.array([1] * 18),
                 R_initial=0,
                 E_initial=np.array([0] * 18),
                 HGen_initial=np.array([0] * 18),
                 HICU_initial=np.array([0] * 18),
                 HICUVent_initial=np.array([0] * 18),
                 birth_rate=0.0003,  # birth rate per capita per day
                 natural_death_rate=1 / (120 * 365),
                 age_bin_edges=np.array([0, 5, 10, 15, 20, 25,
                                         30, 35, 40, 45, 50, 55,
                                         60, 65, 70, 75, 80, 85]),
                 num_compartments_by_age=7,
                 num_compartments_not_by_age=7,
                 max_age=120,
                 D_initial=0,
                 R0=3.75,
                 R0_hospital=0.6,
                 sigma=1 / 5.2,
                 delta=1 / 2.5,
                 delta_hospital=1 / 8.0,
                 kappa=1,
                 gamma=0.5,
                 contact_matrix=np.random.rand(18, 18),
                 approximate_R0=True,
                 # data source: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm#T1_down
                 # rates have been interpolated through centers of age_bin_edges
                 hospitalization_rate_general=np.array(
                     [0.02, 0.02, 0.06, 0.11, 0.15, 0.16,
                      0.18, 0.19, 0.2, 0.23, 0.22, 0.23,
                      0.27, 0.33, 0.33, 0.38, 0.37, 0.51]),
                 hospitalization_rate_icu=np.array(
                     [0.0001, 0.0001, 0.01, 0.02, 0.02, 0.03, 0.03,
                      0.04, 0.05, 0.07, 0.06, 0.07, 0.08,
                      0.11, 0.12, 0.16, 0.13, 0.18]),
                 fraction_icu_requiring_ventilator=0.75,
                 symptoms_to_hospital_days=5,
                 symptoms_to_mortality_days=13,
                 hospitalization_length_of_stay_general=7,
                 hospitalization_length_of_stay_icu=16,
                 hospitalization_length_of_stay_icu_and_ventilator=17,
                 beds_general=300,
                 beds_ICU=100,
                 ventilators=60,
                 # obtained by interpolating through age groups and shift to
                 # get average mortality rate 0.4
                 mortality_rate_from_ICU=np.array([0.373, 0.373, 0.373, 0.374, 0.374,
                                          0.374, 0.375, 0.376, 0.378, 0.379,
                                          0.384, 0.391, 0.397, 0.406, 0.414,
                                          0.433, 0.464, 0.562]),
                 mortality_rate_from_hospital=0.0,
                 mortality_rate_no_ICU_beds=1.,
                 mortality_rate_from_ICUVent=1.0,
                 mortality_rate_no_general_beds=0.0,
                 initial_hospital_bed_utilization=0.6):

        self.N = np.array(N)
        if suppression_policy is None:
            suppression_policy = lambda x: 1
        self.suppression_policy = suppression_policy
        self.I_initial = np.array(I_initial)
        self.A_initial = np.array(A_initial)
        self.R_initial = R_initial
        self.E_initial = np.array(E_initial)
        self.D_initial = D_initial

        self.HGen_initial = np.array(HGen_initial)
        self.HICU_initial = np.array(HICU_initial)
        self.HICUVent_initial = np.array(HICUVent_initial)

        self.S_initial = self.N - self.A_initial - self.I_initial - self.R_initial - self.E_initial \
                         - self.D_initial - self.HGen_initial - self.HICU_initial \
                         - self.HICUVent_initial

        self.birth_rate = birth_rate
        self.natural_death_rate = natural_death_rate

        # Create age steps and groups to define age compartments
        self.age_steps = np.array(age_bin_edges)[1:] - np.array(age_bin_edges)[:-1]
        self.age_steps *= 365  # the model is using day as time unit
        self.age_steps = np.append(self.age_steps, max_age * 365 - age_bin_edges[-1])
        self.age_groups = list(zip(list(age_bin_edges[:-1]), list(age_bin_edges[1:])))
        self.age_groups.append((age_bin_edges[-1], max_age))
        self.num_compartments_by_age = num_compartments_by_age
        self.num_compartments_not_by_age = num_compartments_not_by_age

        # Epidemiological Parameters
        self.R0 = R0                    # Reproduction Number
        self.R0_hospital = R0_hospital  # Reproduction Number at hospital
        self.delta = delta              # 1 / infectious period
        self.delta_hospital = delta_hospital
        self.beta_hospital = self.R0_hospital * self.delta_hospital
        self.sigma = sigma              # Latent Period = 1 / incubation
        self.gamma = gamma              # Clinical outbreak rate
        self.kappa = kappa              # Discount fraction due to isolation of symptomatic cases.

        self.contact_matrix = np.array(contact_matrix)
        self.approximate_R0 = approximate_R0
        self.symptoms_to_hospital_days = symptoms_to_hospital_days
        self.symptoms_to_mortality_days = symptoms_to_mortality_days

        self.hospitalization_rate_general = np.array(hospitalization_rate_general)
        self.hospitalization_rate_icu = np.array(hospitalization_rate_icu)
        self.hospitalization_length_of_stay_general = hospitalization_length_of_stay_general
        self.hospitalization_length_of_stay_icu = hospitalization_length_of_stay_icu
        self.hospitalization_length_of_stay_icu_and_ventilator = hospitalization_length_of_stay_icu_and_ventilator

        self.fraction_icu_requiring_ventilator = fraction_icu_requiring_ventilator

        # Capacity
        self.beds_general = beds_general
        self.beds_ICU = beds_ICU
        self.ventilators = ventilators

        self.mortality_rate_no_general_beds = mortality_rate_no_general_beds
        self.mortality_rate_no_ICU_beds = mortality_rate_no_ICU_beds
        self.mortality_rate_from_ICUVent = mortality_rate_from_ICUVent
        self.initial_hospital_bed_utilization = initial_hospital_bed_utilization

        self.mortality_rate_from_ICU = np.array(mortality_rate_from_ICU)
        self.mortality_rate_from_hospital = mortality_rate_from_hospital

        # beta as the transmission probability per contact times the rescale
        # factor to rescale contact matrix data to match expected R0
        self.beta = self._estimate_beta(self.R0)

        # List of times to integrate.
        self.t_list = t_list
        self.results = None

    def _aging_rate(self, v):
        """
        Calculate rate of aging given compartments size.

        Parameters
        ----------
        v : np.array
            age compartments that correspond to each age group.

        Returns
        -------
        age_in: np.array
            Rate of flow into each compartment in v as result of aging.
        age_out: np.array
            Rate of flow out of each compartment in v as result of aging.
        """

        age_in = v[:-1] / self.age_steps[:-1]
        age_in = np.insert(age_in, 0, 0)
        age_out = v / self.age_steps

        return age_in, age_out

    def calculate_R0(self, beta=None, S_fracs=None):
        """
        Using Next Generation Matrix method to calculate R0 given beta.
        When beta is None, use its default value 1.
        Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2871801/
        The method R0 is calculated as the dominant eigenvalue of next
        generation matrix K, which is the product of transmission matrix T
        and negative inverse of transition matrix Z.
        Transmission matrix T describes rate of immediate new infections in
        state of row caused by state of column. Since all new infections
        first become exposed, values are only nonzero for corresponding rows.
        Transition matrix Z describes rate of flow from column state to row
        state or out of a state (diagonal).

        Parameters
        ----------
        beta: float
            Transmission probability per contact.
        S_fracs: float
            fraction of susceptible population,

        Returns
        -------
        R0 : float
            Basic reproduction number.
        """

        # percentage of susceptible in each age group (assuming that initial
        # condition is disease-free equilibrium)
        if S_fracs is None:
            S_fracs = self.N / self.N.sum()

        beta = beta or 1
        age_group_num = list(self.N.shape)[0]
        # contact with susceptible at disease-free equilibrium
        # [C_11 * P_1, C_12 * P_1, ... C_1n * P_n]
        # ...
        # [C_n1 * P_n, C_n2 * P_n, ... C_nn * P_n]
        contact_with_susceptible = self.contact_matrix * S_fracs.T

        # transmission matrix with rates of immediate new infections into
        # compartment of rows due to transmission from compartment from
        # columns.
        # All sub_matrix is named as T_<row compartment>_<column compartment>
        T_E_E = np.zeros((age_group_num, age_group_num))           # E to E
        T_E_A = contact_with_susceptible * beta                    # A to E
        T_E_I = contact_with_susceptible * beta * self.kappa       # I to E
        T_E_nonICU = contact_with_susceptible * self.beta_hospital # nonICU to E
        T_E_ICU = contact_with_susceptible * self.beta_hospital    # ICU to E

        # all immediate new infections flow into E so the rest values are
        # zeros
        T_A_E = np.zeros((age_group_num, age_group_num))      # E to A
        T_A_A = np.zeros((age_group_num, age_group_num))      # A to A
        T_A_I = np.zeros((age_group_num, age_group_num))      # I to A
        T_A_nonICU = np.zeros((age_group_num, age_group_num)) # nonICU to A
        T_A_ICU = np.zeros((age_group_num, age_group_num))    # ICU to A

        T_I_E = np.zeros((age_group_num, age_group_num))      # E to I
        T_I_A = np.zeros((age_group_num, age_group_num))      # A to I
        T_I_I = np.zeros((age_group_num, age_group_num))      # I to I
        T_I_nonICU = np.zeros((age_group_num, age_group_num)) # nonICU to I
        T_I_ICU = np.zeros((age_group_num, age_group_num))    # ICU to I

        T_nonICU_E = np.zeros((age_group_num, age_group_num)) # E to nonICU
        T_nonICU_A = np.zeros((age_group_num, age_group_num)) # A to nonICU
        T_nonICU_I = np.zeros((age_group_num, age_group_num)) # I to nonICU
        T_nonICU_nonICU = np.zeros((age_group_num, age_group_num)) # nonICU TO nonICU
        T_nonICU_ICU = np.zeros((age_group_num, age_group_num))    # ICU to nonICU

        T_ICU_E = np.zeros((age_group_num, age_group_num))  # E to nonICU
        T_ICU_A = np.zeros((age_group_num, age_group_num))  # A to nonICU
        T_ICU_I = np.zeros((age_group_num, age_group_num))  # I to nonICU
        T_ICU_nonICU = np.zeros((age_group_num, age_group_num))  # nonICU TO nonICU
        T_ICU_ICU = np.zeros((age_group_num, age_group_num))  # ICU to nonICU

        T_E = np.concatenate([T_E_E, T_E_A, T_E_I, T_E_nonICU, T_E_ICU], axis=1)  # all rates to E
        T_A = np.concatenate([T_A_E, T_A_A, T_A_I, T_A_nonICU, T_A_ICU], axis=1)  # all rates to A
        T_I = np.concatenate([T_I_E, T_I_A, T_I_I, T_I_nonICU, T_I_ICU], axis=1)  # all rates to I
        T_nonICU = np.concatenate([T_nonICU_E, T_nonICU_A, T_nonICU_I,
                                   T_nonICU_nonICU, T_nonICU_ICU], axis=1)  # all rates to nonICU
        T_ICU = np.concatenate([T_ICU_E, T_ICU_A, T_ICU_I,
                                T_ICU_nonICU, T_ICU_ICU], axis=1)  # all rates to ICU
        T = np.concatenate([T_E, T_A, T_I, T_nonICU, T_ICU])


        # Matrix of rates of transitions from compartment of rows to
        # compartments of columns, or out of a compartment (when row = column)
        # All sub_matrix is named as Z_<row compartment>_<column compartment>

        # rates of transition out of E (incubation or aging) and into E (aging)
        aging_rate_in = 1 / self.age_steps[:-1]
        aging_rate_out = 1 / self.age_steps
        Z_E_E = np.diag(-(aging_rate_out + self.sigma)) + np.diag(aging_rate_in, k=-1)
        Z_E_A = np.zeros((age_group_num, age_group_num))
        Z_E_I = np.zeros((age_group_num, age_group_num))
        Z_E_nonICU = np.zeros((age_group_num, age_group_num))
        Z_E_ICU = np.zeros((age_group_num, age_group_num))
        Z_E = np.concatenate([Z_E_E, Z_E_A, Z_E_I, Z_E_nonICU, Z_E_ICU], axis=1)

        # rates of transition out of A (recovery and aging) and into A (aging
        # and from exposed)
        Z_A_E = np.zeros((age_group_num, age_group_num))
        np.fill_diagonal(Z_A_E, self.sigma * (1 - self.gamma))   # from E to A
        Z_A_A = np.diag(-(aging_rate_out + self.delta)) + np.diag(aging_rate_in, k=-1)
        Z_A_I = np.zeros((age_group_num, age_group_num))
        Z_A_nonICU = np.zeros((age_group_num, age_group_num))
        Z_A_ICU = np.zeros((age_group_num, age_group_num))
        Z_A = np.concatenate([Z_A_E, Z_A_A, Z_A_I, Z_A_nonICU, Z_A_ICU], axis=1)

        # rates of transition out of I (recovery, hospitalization and aging)
        # and into I (aging and from exposed)
        rate_recovered = self.delta
        rate_in_hospital_general = (
                self.hospitalization_rate_general - self.hospitalization_rate_icu) / self.symptoms_to_hospital_days
        rate_in_hospital_icu = self.hospitalization_rate_icu / self.symptoms_to_hospital_days
        rate_out_of_I = aging_rate_out + rate_recovered + rate_in_hospital_general + rate_in_hospital_icu

        Z_I_E = np.zeros((age_group_num, age_group_num))
        np.fill_diagonal(Z_I_E, self.sigma * self.gamma)  # from E to I
        Z_I_A = np.zeros((age_group_num, age_group_num))
        Z_I_I = np.diag(-rate_out_of_I) + np.diag(aging_rate_in, k=-1)  # transition out of I
        Z_I_nonICU = np.zeros((age_group_num, age_group_num))
        Z_I_ICU = np.zeros((age_group_num, age_group_num))
        Z_I = np.concatenate([Z_I_E, Z_I_A, Z_I_I, Z_I_nonICU, Z_I_ICU], axis=1)

        # rates of transition out of nonICU (recovery, death and aging)
        # and into nonICU (aging and from infected)
        died_from_hosp = self.mortality_rate_from_hospital / \
                         self.hospitalization_length_of_stay_general
        recovered_after_hospital_general = (1 - self.mortality_rate_from_hospital) / \
                                           self.hospitalization_length_of_stay_general
        rate_out_of_nonICU = aging_rate_out + died_from_hosp + recovered_after_hospital_general

        Z_nonICU_E = np.zeros((age_group_num, age_group_num))
        Z_nonICU_A = np.zeros((age_group_num, age_group_num))
        Z_nonICU_I = np.diag(rate_in_hospital_general)
        Z_nonICU_nonICU = np.diag(-(rate_out_of_nonICU)) + np.diag(aging_rate_in, k=-1)
        Z_nonICU_ICU = np.zeros((age_group_num, age_group_num))
        Z_nonICU = np.concatenate([Z_nonICU_E, Z_nonICU_A, Z_nonICU_I, Z_nonICU_nonICU, Z_nonICU_ICU], axis=1)

        # rates of transition out of ICU (recovery, death and aging) and into
        # ICU (aging and from infected)
        died_from_icu = (1 - self.fraction_icu_requiring_ventilator) * self.mortality_rate_from_ICU / \
                        self.hospitalization_length_of_stay_icu
        died_from_icu_vent = self.mortality_rate_from_ICUVent / self.hospitalization_length_of_stay_icu_and_ventilator
        recovered_from_icu_no_vent = (1 - self.mortality_rate_from_ICU) * (1 - self.fraction_icu_requiring_ventilator) \
                                     / self.hospitalization_length_of_stay_icu
        recovered_from_icu_vent = (1 - np.maximum(self.mortality_rate_from_ICU, self.mortality_rate_from_ICUVent))\
                                  / self.hospitalization_length_of_stay_icu_and_ventilator
        rate_out_of_ICU = aging_rate_out + died_from_icu + died_from_icu_vent + recovered_from_icu_no_vent + recovered_from_icu_vent

        Z_ICU_E = np.zeros((age_group_num, age_group_num))
        Z_ICU_A = np.zeros((age_group_num, age_group_num))
        Z_ICU_I = np.diag(rate_in_hospital_icu)
        Z_ICU_nonICU = np.zeros((age_group_num, age_group_num))
        Z_ICU_ICU = np.diag(-(rate_out_of_ICU)) + np.diag(aging_rate_in, k=-1)
        Z_ICU = np.concatenate([Z_ICU_E, Z_ICU_A, Z_ICU_I, Z_ICU_nonICU, Z_ICU_ICU], axis=1)
        Z = np.concatenate([Z_E, Z_A, Z_I, Z_nonICU, Z_ICU])

        # Calculate R0 from transmission and transition matrix
        Z_inverse = np.linalg.inv(Z)
        K = T.dot(-Z_inverse)
        eigen_values = np.linalg.eigvals(K)
        R0 = max(eigen_values)  # R0 is the dominant eigenvalue

        return R0

    def _estimate_beta(self, expected_R0):
        """
        Estimate beta for given R0. Note that beta can be greater than 1 for
        some expected R0 given a contact matrix. This is because the
        contact matrix sometimes underestimate overall contact rate.
        In this case, beta is the product of transmission probability per
        contact and the factor that lift the average contact rate to match
        the expected R0.

        Parameters
        ----------
        expected_R0 : float
            R0 to solve for beta.

        Returns
        -------
          : float
            Beta that give rise the expected R0.
        """
        # transmission matrix with rates of immediate new infections into rows
        # due to transmission from columns
        R0 = self.calculate_R0()
        beta = expected_R0 / R0 # R0 linearly increases as beta increases
        return float(beta.real)

    def calculate_Rt(self, S_fracs, suppression_policy=None):
        """
        Calculate R(t)

        Parameters
        ----------
        S_fracs: np.array
            Fraction of each age group among susceptible population.
        suppression_policy: int or np.array
            Fraction of remained effective contacts as result suppression
            policy through time.

        Returns
        -------
        Rt: np.array
            Basic reproduction number through time.
        """
        Rt = np.zeros(S_fracs.shape[1])
        for n in range(S_fracs.shape[1]):
            Rt[n] += self.calculate_R0(self.beta, S_fracs[:, n])
        Rt *= suppression_policy
        return Rt

    def _time_step(self, t, y):
        """
        One integral moment.

        Parameters
        ----------
        y: array
            Input compartment size
        t: float
            Time step.

        Returns
        -------
          :  np.array
            ODE derivatives.
        """
        # np.split(y[:-7], 7)  <--- This is 7x slower than the code below.
        chunk_size = y[:-self.num_compartments_not_by_age].shape[0] // self.num_compartments_by_age
        S, E, A, I, HNonICU, HICU, HICUVent = [y[(i * chunk_size):((i + 1) * chunk_size)]
                                               for i in range(self.num_compartments_by_age)]

        R = y[-7]

        # TODO: County-by-county affinity matrix terms can be used to describe
        # transmission network effects. ( also known as Multi-Region SEIR)
        # https://arxiv.org/pdf/2003.09875.pdf
        #  For those living in county i, the interacting county j exposure is given
        #  by A term dE_i/dt += N_i * Sum_j [ beta_j * mix_ij * I_j * S_i + beta_i *
        #  mix_ji * I_j * S_i ] mix_ij can be proxied by Census-based commuting
        #  matrices as workplace interactions are the dominant term. See:
        #  https://www.census.gov/topics/employment/commuting/guidance/flows.html
        # Effective contact rate * those that get exposed * those susceptible.
        total_ppl = self.N.sum()

        # get matrix:
        # [C_11 * S_1 * I_1/N, ... C1j * S_1 * I_j/N...]
        # [C_21 * S_2 * I_1/N, ... C1j * S_2 * I_j/N...]
        # ...
        # [C_21 * S_n * I_1/N, ... C_21 * S_n * I_j/N ...]
        frac_infected = (self.kappa * I + A) / total_ppl
        frac_hospt = (HICU + HNonICU) / total_ppl
        S_and_I = S[:, np.newaxis].dot(frac_infected[:, np.newaxis].T)
        S_and_hosp = S[:, np.newaxis].dot(frac_hospt[:, np.newaxis].T)
        contacts_S_and_I = (S_and_I * self.contact_matrix).sum(axis=1)
        contacts_S_and_hosp = (S_and_hosp * self.contact_matrix).sum(axis=1)
        number_exposed = self.beta * self.suppression_policy(t) * contacts_S_and_I + self.beta_hospital * contacts_S_and_hosp

        age_in_S, age_out_S = self._aging_rate(S)
        age_in_S[0] = self.N.sum() * self.birth_rate
        dSdt = age_in_S - number_exposed - age_out_S

        exposed_and_symptomatic = self.gamma * self.sigma * E           # latent period moving to infection = 1 / incubation
        exposed_and_asymptomatic = (1 - self.gamma) * self.sigma * E    # latent period moving to asymptomatic but infected) = 1 / incubation
        age_in_E, age_out_E = self._aging_rate(E)
        dEdt = age_in_E + number_exposed - exposed_and_symptomatic - exposed_and_asymptomatic - age_out_E

        asymptomatic_and_recovered = self.delta * A
        age_in_A, age_out_A = self._aging_rate(A)
        dAdt = age_in_A + exposed_and_asymptomatic - asymptomatic_and_recovered - age_out_A

        # Fraction that didn't die or go to hospital
        infected_and_recovered_no_hospital = self.delta * I
        infected_and_in_hospital_general = I * (
                    self.hospitalization_rate_general - self.hospitalization_rate_icu) / self.symptoms_to_hospital_days
        infected_and_in_hospital_icu = I * self.hospitalization_rate_icu / self.symptoms_to_hospital_days

        age_in_I, age_out_I = self._aging_rate(I)
        dIdt = age_in_I \
             + exposed_and_symptomatic \
             - infected_and_recovered_no_hospital \
             - infected_and_in_hospital_general \
             - infected_and_in_hospital_icu \
             - age_out_I

        mortality_rate_ICU = self.mortality_rate_from_ICU if sum(HICU) <= self.beds_ICU else self.mortality_rate_no_ICU_beds
        mortality_rate_NonICU = self.mortality_rate_from_hospital if sum(HNonICU) <= self.beds_general else \
            self.mortality_rate_no_general_beds

        died_from_hosp = HNonICU * mortality_rate_NonICU / self.hospitalization_length_of_stay_general
        died_from_icu = HICU * (
                1 - self.fraction_icu_requiring_ventilator) * mortality_rate_ICU / \
                        self.hospitalization_length_of_stay_icu
        died_from_icu_vent = HICUVent * self.mortality_rate_from_ICUVent / \
                             self.hospitalization_length_of_stay_icu_and_ventilator

        recovered_after_hospital_general = \
            HNonICU * (1 - mortality_rate_NonICU) / self.hospitalization_length_of_stay_general
        recovered_from_icu_no_vent = \
            HICU * (1 - mortality_rate_ICU) * (1 - self.fraction_icu_requiring_ventilator) \
            / self.hospitalization_length_of_stay_icu
        recovered_from_icu_vent = \
            HICUVent * (1 - np.maximum(mortality_rate_ICU, self.mortality_rate_from_ICUVent)) \
            / self.hospitalization_length_of_stay_icu_and_ventilator

        age_in_HNonICU, age_out_HNonICU = self._aging_rate(HNonICU)
        dHNonICU_dt = age_in_HNonICU + infected_and_in_hospital_general - recovered_after_hospital_general - \
                      died_from_hosp - age_out_HNonICU

        age_in_HICU, age_out_HICU = self._aging_rate(HICU)
        dHICU_dt = (age_in_HICU
                   + infected_and_in_hospital_icu
                   - recovered_from_icu_no_vent
                   - recovered_from_icu_vent
                   - died_from_icu
                   - died_from_icu_vent
                   - age_out_HICU)

        # This compartment is for tracking ventillator count. The beds are
        # accounted for in the ICU cases.
        age_in_HICUVent, age_out_HICUVent = self._aging_rate(HICUVent)
        rate_ventilator_needed = infected_and_in_hospital_icu * self.fraction_icu_requiring_ventilator
        rate_removing_ventilator = HICUVent / self.hospitalization_length_of_stay_icu_and_ventilator
        dHICUVent_dt = age_in_HICUVent + rate_ventilator_needed - \
                       rate_removing_ventilator - age_out_HICUVent

        # Tracking categories...
        dTotalInfections = sum(exposed_and_symptomatic) + sum(exposed_and_asymptomatic)
        dHAdmissions_general = sum(infected_and_in_hospital_general)
        dHAdmissions_ICU = sum(infected_and_in_hospital_icu)  # Ventilators also count as ICU beds.

        # Fraction that recover
        dRdt = (sum(asymptomatic_and_recovered)
              + sum(infected_and_recovered_no_hospital)
              + sum(recovered_after_hospital_general)
              + sum(recovered_from_icu_vent)
              + sum(recovered_from_icu_no_vent)
              - R * self.natural_death_rate)

        # Death among hospitalized.
        dDdt = sum(died_from_icu) + sum(died_from_icu_vent) + sum(died_from_hosp)
        died_from_hospital_bed_limits = max(sum(HNonICU) - self.beds_general, 0) * self.mortality_rate_no_general_beds \
                                            / self.hospitalization_length_of_stay_general
        died_from_icu_bed_limits = max(sum(HICU) - self.beds_ICU, 0) * self.mortality_rate_no_ICU_beds \
                                       / self.hospitalization_length_of_stay_icu

        # death due to hospital bed limitation
        dD_no_hgendt = died_from_hospital_bed_limits
        dD_no_icudt = died_from_icu_bed_limits

        return np.concatenate([dSdt, dEdt, dAdt, dIdt, dHNonICU_dt, dHICU_dt, dHICUVent_dt,
                               np.array([dRdt, dDdt, dD_no_hgendt, dD_no_icudt,
                                         dHAdmissions_general, dHAdmissions_ICU,
                                         dTotalInfections])])

    def run(self):
        """
        Integrate the ODE numerically.

        Returns
        -------
        results: dict
        {
            't_list': self.t_list,
            'S': susceptible population combining all age groups,
            'E': exposed population combining all age groups,
            'I': symptomatic population combining all age groups,
            'A': asymptomatic population combining all age groups,
            'R': recovered population,
            'HGen': general hospitalized population combining all age groups,
            'HICU': icu admitted population combining all age groups,
            'HVent': population on ventilator combining all age groups,
            'D': Deaths during hospitalization,
            'deaths_from_hospital_bed_limits': deaths due to hospital bed
                                               limitation
            'deaths_from_icu_bed_limits': deaths due to icu limitation
            'deaths_from_ventilator_limits': deaths due to ventilator limitation
            'total_deaths': Deaths
            'by_age':
                {
                    'S': susceptible population by age group
                    'E': exposed population by age group
                    'I': symptomatic population by age group
                    'A': asymptomatic population by age group
                    'HGen': general hospitalized population by age group
                    'HICU': icu admitted population by age group
                    'HVent': population on ventilator by age group
        }
        """
        # Initial conditions vector
        D_no_hgen, D_no_icu, HAdmissions_general, HAdmissions_ICU, TotalAllInfections = 0, 0, 0, 0, 0
        y0 = np.concatenate([self.S_initial, self.E_initial, self.A_initial, self.I_initial,
                             self.HGen_initial, self.HICU_initial, self.HICUVent_initial,
                             np.array([self.R_initial, self.D_initial,
                                       D_no_hgen, D_no_icu,
                                       HAdmissions_general, HAdmissions_ICU,
                                       TotalAllInfections])])

        # Integrate the SEIR equations over the time grid, t.
        result_time_series = solve_ivp(fun=self._time_step,
                                       t_span=[self.t_list.min(), self.t_list.max()],
                                       y0=y0,
                                       t_eval=self.t_list,
                                       method='RK23', rtol=1e-3, atol=1e-3).y

        S, E, A, I, HGen, HICU, HICUVent = np.split(result_time_series[:-self.num_compartments_not_by_age],
                                                    self.num_compartments_by_age)
        R, D, D_no_hgen, D_no_icu, HAdmissions_general, HAdmissions_ICU, TotalAllInfections = result_time_series[-7:]

        if self.approximate_R0:
            Rt = np.zeros(len(self.t_list))
            Rt += self.R0 * self.suppression_policy(self.t_list)
        else:
            S_fracs_within_age_group = S / S.sum(axis=0)
            Rt = self.calculate_Rt(S_fracs_within_age_group, self.suppression_policy(self.t_list))

        self.results = {
            't_list': self.t_list,
            'S': S.sum(axis=0),
            'E': E.sum(axis=0),
            'A': A.sum(axis=0),
            'I': I.sum(axis=0),
            'R': R,
            'HGen': HGen.sum(axis=0),
            'HICU': HICU.sum(axis=0),
            'HVent': HICUVent.sum(axis=0),
            'D': D,
            'Rt': Rt,
            'direct_deaths_per_day': np.array([0] + list(np.diff(D))), # Derivative...
            # Here we assume that the number of person days above the saturation
            # divided by the mean length of stay approximates the number of
            # deaths from each source.
            # Ideally this is included in the dynamics, but this is left as a TODO.
            'deaths_from_hospital_bed_limits': D_no_hgen,
            # Here ICU = ICU + ICUVent, but we want to remove the ventilated fraction and account for that below.
            'deaths_from_icu_bed_limits': D_no_icu,
            'HGen_cumulative': np.cumsum(HGen.sum(axis=0)) / self.hospitalization_length_of_stay_general,
            'HICU_cumulative': np.cumsum(HICU.sum(axis=0)) / self.hospitalization_length_of_stay_icu,
            'HVent_cumulative': np.cumsum(HICUVent.sum(axis=0)) / self.hospitalization_length_of_stay_icu_and_ventilator
        }

        self.results['total_deaths'] = D + D_no_hgen + D_no_icu

        # Derivatives of the cumulative give the "new" infections per day.
        self.results['total_new_infections'] = np.append([0], np.diff(TotalAllInfections))
        self.results['total_deaths_per_day'] = np.append([0], np.diff(self.results['total_deaths']))
        self.results['general_admissions_per_day'] = np.append([0], np.diff(HAdmissions_general))
        self.results['icu_admissions_per_day'] = np.append([0], np.diff(HAdmissions_ICU))  # Derivative of the
        # cumulative.

        self.results['by_age'] = dict()
        self.results['by_age']['S'] = S
        self.results['by_age']['E'] = E
        self.results['by_age']['A'] = A
        self.results['by_age']['I'] = I
        self.results['by_age']['HGen'] = HGen
        self.results['by_age']['HICU'] = HICU
        self.results['by_age']['HVent'] = HICUVent

    def plot_results(self, y_scale='log', by_age_group=False, xlim=None):
        """
        Generate a summary plot for the simulation.

        Parameters
        ----------
        y_scale: str
            Matplotlib scale to use on y-axis. Typically 'log' or 'linear'
        by_age_group: bool
            Whether plot projections by age group.
        xlim: float
            Limits of x axis.
        """
        if not by_age_group:
            # Plot the data on three separate curves for S(t), I(t) and R(t)
            fig = plt.figure(facecolor='w', figsize=(10, 8))
            plt.subplot(221)
            plt.plot(self.t_list, self.results['S'], alpha=1, lw=2, label='Susceptible')
            plt.plot(self.t_list, self.results['E'], alpha=.5, lw=2, label='Exposed')
            plt.plot(self.t_list, self.results['A'], alpha=.5, lw=2, label='Asymptomatic')
            plt.plot(self.t_list, self.results['I'], alpha=.5, lw=2, label='Infected')
            plt.plot(self.t_list, self.results['R'], alpha=1, lw=2, label='Recovered & Immune', linestyle='--')

            plt.plot(self.t_list, self.results['S'] + self.results['E']
                     + self.results['A'] + self.results['I']
                     + self.results['R'] + self.results['D']
                     + self.results['HGen'] + self.results['HICU'], label='Total')

            plt.xlabel('Time [days]', fontsize=12)
            plt.yscale(y_scale)
            # plt.ylim(1, plt.ylim(1))
            plt.grid(True, which='both', alpha=.35)
            plt.legend(framealpha=.5)
            if xlim:
                plt.xlim(*xlim)
            else:
                plt.xlim(0, self.t_list.max())
            plt.ylim(1, self.N.sum(axis=0) * 1.1)

            plt.subplot(222)

            plt.plot(self.t_list, self.results['D'], alpha=.4, c='k', lw=1, label='Direct Deaths',
                     linestyle='-')

            plt.plot(self.t_list, self.results['HGen'], alpha=1, lw=2, c='steelblue',
                     label='General Beds Required', linestyle='-')
            plt.hlines(self.beds_general, self.t_list[0], self.t_list[-1], 'steelblue', alpha=1, lw=2, label='ICU Bed Capacity', linestyle='--')

            plt.plot(self.t_list, self.results['HICU'], alpha=1, lw=2, c='firebrick', label='ICU Beds Required', linestyle='-')
            plt.hlines(self.beds_ICU, self.t_list[0], self.t_list[-1], 'firebrick', alpha=1, lw=2, label='General Bed Capacity', linestyle='--')

            plt.plot(self.t_list, self.results['HVent'], alpha=1, lw=2, c='seagreen', label='Ventilators Required', linestyle='-')
            plt.hlines(self.ventilators, self.t_list[0], self.t_list[-1], 'seagreen', alpha=1, lw=2, label='Ventilator Capacity', linestyle='--')

            plt.xlabel('Time [days]', fontsize=12)
            plt.ylabel('')
            plt.yscale(y_scale)
            plt.ylim(1, plt.ylim()[1])
            plt.grid(True, which='both', alpha=.35)
            plt.legend(framealpha=.5)
            if xlim:
                plt.xlim(*xlim)
            else:
                plt.xlim(0, self.t_list.max())

            plt.subplot(223)
            plt.plot(self.t_list, [self.suppression_policy(t) for t in self.t_list], c='steelblue')
            plt.ylabel('Contact Rate Reduction')
            plt.xlabel('Time [days]', fontsize=12)
            plt.grid(True, which='both')

            # Reproduction number through time
            plt.subplot(224)
            plt.plot(self.t_list, self.results['Rt'], c='steelblue')
            plt.ylabel('R(t)')
            plt.xlabel('Time [days]', fontsize=12)
            plt.grid(True, which='both')

            plt.tight_layout()

        else:
            # Plot the data by age group
            fig, axes = plt.subplots(len(self.age_groups), 2, figsize=(10, 50))
            for ax, n in zip(axes, range(len(self.age_groups))):
                ax1, ax2 = ax
                ax1.plot(self.t_list, self.results['by_age']['S'][n, :], alpha=1, lw=2, label='Susceptible')
                ax1.plot(self.t_list, self.results['by_age']['E'][n, :], alpha=.5, lw=2, label='Exposed')
                ax1.plot(self.t_list, self.results['by_age']['A'][n, :], alpha=.5, lw=2, label='Asymptomatic')
                ax1.plot(self.t_list, self.results['by_age']['I'][n, :], alpha=.5, lw=2, label='Infected')
                ax2.plot(self.t_list, self.results['by_age']['HGen'][n, :], alpha=1, lw=2, label='Hospital general',
                         linestyle='--')
                ax2.plot(self.t_list, self.results['by_age']['HICU'][n, :], alpha=1, lw=2, label='ICU',
                         linestyle='--')
                ax2.plot(self.t_list, self.results['by_age']['HVent'][n, :], alpha=1, lw=2, label='ICUVent',
                         linestyle='--')
                ax1.legend()
                ax2.legend()
                ax1.set_xlabel('days')
                ax2.set_xlabel('days')
                ax1.set_title('age group %d-%d' %(self.age_groups[n][0],
                                                  self.age_groups[n][1]))
                ax1.set_yscale('log')
                ax2.set_yscale('log')
                ax1.set_ylim(ymin=1)
                ax2.set_ylim(ymin=1)

            plt.tight_layout()

        return fig
