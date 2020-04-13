import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy


class SEIRModelAge:

    def __init__(self,
                 N,
                 t_list,
                 suppression_policy,
                 A_initial=np.array([1] * 14),
                 I_initial=np.array([1] * 14),
                 R_initial=np.array([0] * 14),
                 E_initial=np.array([0] * 14),
                 HGen_initial=np.array([0] * 14),
                 HICU_initial=np.array([0] * 14),
                 HICUVent_initial=np.array([0] * 14),
                 birth_rate=0.0003,  # birth rate per capita per day
                 age_bin_edges=np.array([0, 5, 10, 15, 20, 25,
                                         30, 35, 40, 45, 50, 55,
                                         60, 65, 70, 75, 80, 85]),
                 D_initial=np.array([0] * 14),
                 R0=3.75,
                 sigma=1 / 5.2,
                 delta=1 / 2.5,
                 kappa=1,
                 gamma=0.5,
                 contact_matrix=np.random.rand(5, 5),
                 # data source: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm#T1_down
                 # rates have been interpolated through centers of age_bin_edges
                 hospitalization_rate_general=[0.02, 0.02, 0.06, 0.11, 0.15, 0.16,
                                               0.18, 0.19, 0.2, 0.23, 0.22, 0.23,
                                               0.27, 0.33, 0.33, 0.38, 0.37, 0.51],
                 hospitalization_rate_icu=[0.0, 0.0, 0.01, 0.02, 0.02, 0.03, 0.03,
                                           0.04, 0.05, 0.07, 0.06, 0.07, 0.08,
                                           0.11, 0.12, 0.16, 0.13, 0.18],
                 fraction_icu_requiring_ventilator=0.75,
                 symptoms_to_hospital_days=5,
                 symptoms_to_mortality_days=13,
                 hospitalization_length_of_stay_general=7,
                 hospitalization_length_of_stay_icu=16,
                 hospitalization_length_of_stay_icu_and_ventilator=17,
                 beds_general=300,
                 beds_ICU=100,
                 ventilators=60,
                 mortality_rate=[0.0, 0.0, 0.00029, 0.00076, 0.0011, 0.00131,
                                 0.00163, 0.00298, 0.00433, 0.00583, 0.01059,
                                 0.01733, 0.02382, 0.03311, 0.04073, 0.06022,
                                 0.09036, 0.1885],
                 mortality_rate_no_ICU_beds=0.85,
                 mortality_rate_no_ventilator=1.0,
                 mortality_rate_no_general_beds=0.6):
        """
        This class implements a SEIR-like compartmental epidemic model
        consisting of SEIR states plus death, and hospitalizations.

        In the diff eq modeling, these parameters are assumed exponentially
        distributed and modeling occurs in the thermodynamic limit, i.e. we do
        not perform monte carlo for individual cases.

        Model Refs:
         # TODO: Incorporate structures and Parameters from Weitz group
         - https://github.com/jsweitz/covid-19-ga-summer-2020/blob/master/fignearterm_0328_alt.m
            ```
                % Init the population - baseline
                % Open plus hospitals
                % SEIaIS (open) and then I_ha I_hs and then R (open) and D (cumulative) age stratified
                tmpzeros = zeros(size(agepars.meanage));
                outbreak.y0=[population.agefrac tmpzeros tmpzeros tmpzeros tmpzeros tmpzeros tmpzeros tmpzeros tmpzeros];
                % Initiate an outbreak with 500 symptomatic current caseas and 7500 asymptomatic cases
                % effective 8000 total and 25 deaths (based on GA estimates)
                % Initiate an outbreak
                pars.alpha=0;  % Shielding
                pars.beta_a=4/10;   % Transmission for asymptomatic
                pars.beta_s=8/10;      % Transmission for symptomatic

                pars.Ra=pars.beta_a/pars.gamma_a;
                pars.Rs=pars.beta_s/pars.gamma_s;
                pars.R0=pars.p*pars.Ra+(1-pars.p)*pars.Rs;
                pars.p=[0.95 0.95 0.90 0.8 0.7 0.6 0.4 0.2 0.2 0.2];         % Fraction asymptomatic by age

                pars.gamma_e=1/4;   % Transition to infectiousness
                pars.gamma_a=1/6;   % Resolution rate for asymptomatic
                pars.gamma_s=1/6;  % Resolution rate for symptomatic
                pars.gamma_h=1/10;  % Resolution rate in hospitals
                pars.beta_a=4/10;   % Transmission for asymptomatic
                pars.beta_s=8/10;      % Transmission for symptomatic


                agepars.meanage=5:10:95;
                agepars.highage=[9:10:99];  % Age groups
                agepars.lowage=[0:10:90];  % Age groups
                agepars.hosp_frac=[0.1 0.3 1.2 3.2 4.9 10.2 16.6 24.3 27.3 27.3]/100;
                agepars.hosp_crit=[5 5 5 5 6.3 12.2 27.4 43.2 70.9 70.9]/100;
                agepars.crit_die= 0.5*ones(size(agepars.meanage));
                agepars.num_ages = length(agepars.meanage);


                % Assign things
                dydt=zeros(length(y),1);
                Ia=sum(y(agepars.Ia_ids));
                Is=sum(y(agepars.Is_ids));
                R = sum(y(agepars.R_ids));
                S = sum(y(agepars.S_ids));
                E = sum(y(agepars.E_ids));

                % Dynamics -  Base Model
                dydt(agepars.S_ids)=-pars.beta_a*y(agepars.S_ids)*Ia-pars.beta_s*y(agepars.S_ids)*Is;
                dydt(agepars.E_ids)=pars.beta_a*y(agepars.S_ids)*Ia+pars.beta_s*y(agepars.S_ids)*Is-pars.gamma_e*y(agepars.E_ids);
                dydt(agepars.Ia_ids)=pars.p'.*pars.gamma_e.*y(agepars.E_ids)-pars.gamma_a*y(agepars.Ia_ids);
                dydt(agepars.Is_ids)=(ones(size(pars.p))-pars.p)'.*pars.gamma_e.*y(agepars.E_ids)-pars.gamma_s*y(agepars.Is_ids);
                dydt(agepars.Ihsub_ids)=agepars.hosp_frac'.*(1-agepars.hosp_crit')*pars.gamma_s.*y(agepars.Is_ids)-pars.gamma_h*y(agepars.Ihsub_ids);
                dydt(agepars.Ihcri_ids)=agepars.hosp_frac'.*agepars.hosp_crit'*pars.gamma_s.*y(agepars.Is_ids)-pars.gamma_h*y(agepars.Ihcri_ids);
                dydt(agepars.R_ids)=pars.gamma_a*y(agepars.Ia_ids)+pars.gamma_s*y(agepars.Is_ids).*(1-agepars.hosp_frac')+pars.gamma_h*y(agepars.Ihsub_ids)+pars.gamma_h*y(agepars.Ihcri_ids).*(1-agepars.crit_die');
                dydt(agepars.D_ids)=pars.gamma_h*y(agepars.Ihcri_ids).*agepars.crit_die';
                dydt(agepars.Hcum_ids)=agepars.hosp_frac'.*(1-agepars.hosp_crit')*pars.gamma_s.*y(agepars.Is_ids)+agepars.hosp_frac'.*agepars.hosp_crit'*pars.gamma_s.*y(agepars.Is_ids);
            ```


         - https://arxiv.org/pdf/2003.10047.pdf  # We mostly follow this notation.
         - https://arxiv.org/pdf/2002.06563.pdf

        Need more details on hospitalization parameters...

        Imperial college has more pessimistic numbers.
        1. https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf

        UW tends to have more optimistic numbers
        2. http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf

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
        birth_rate : float
            Birth per capita per day
        age_steps : np.array
            Time people spend in each age group. Last age bin edge is assumed to be 100 years old.
        age_groups : np.array
            Age groups.
        R0: float
            Basic Reproduction number
        kappa: float
            Fractional contact rate for those with symptoms since they should be
            isolated vs asymptomatic who are less isolated. A value 1 implies
            the same rate. A value 0 implies symptomatic people never infect
            others.
        sigma: float
            Latent decay scale is defined as 1 / incubation period.
            1 / 4.8: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf
            1 / 5.2 [3, 8]: https://arxiv.org/pdf/2003.10047.pdf
        delta : float
            1 / infectious period.
        gamma: float
            Clinical outbreak rate (fraction of infected that show symptoms)
        contact_matrix : pd.DataFrame
            With cell at ith row and jth column as contact rate made by ith age group with jth age group.
        hospitalization_rate_general: float
            Fraction of infected that are hospitalized generally (not in ICU)
            TODO: Make this age dependent
        hospitalization_rate_icu: float
            Fraction of infected that are hospitalized in the ICU
            TODO: Make this age dependent
        hospitalization_length_of_stay_icu_and_ventilator: float
            Mean LOS for those requiring ventilators
        contact_matrix
        fraction_icu_requiring_ventilator: float
            Of the ICU cases, which require ventilators.
        mortality_rate: float
            Fraction of infected that die.
            0.0052: https://arxiv.org/abs/2003.10720
            TODO: Make this age dependent
            TODO: This is modeled below as P(mortality | symptoms) which is higher than the overall mortality rate by factor 2.
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
        """
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

        self.S_initial = self.N - self.A_initial - self.I_initial - self.R_initial - self.E_initial \
                         - self.D_initial - self.HGen_initial - self.HICU_initial \
                         - self.HICUVent_initial

        self.birth_rate = birth_rate

        # Create age steps and groups to define age compartments
        self.age_steps = np.array(age_bin_edges)[1:] - np.array(age_bin_edges)[:-1]
        self.age_steps *= 365  # the model is using day as time unit
        self.age_steps = np.append(self.age_steps, 100 * 365 - age_bin_edges[-1])
        self.age_groups = list(zip(list(age_bin_edges[:-1]), list(age_bin_edges[1:])))
        self.age_groups.append((age_bin_edges[-1], 100))

        # Epidemiological Parameters
        self.R0 = R0              # Reproduction Number
        self.sigma = sigma        # Latent Period = 1 / incubation
        self.delta = delta
        self.gamma = gamma        # Clinical outbreak rate
        self.kappa = kappa        # Discount fraction due to isolation of symptomatic cases.

        self.contact_matrix = contact_matrix
        self.mortality_rate = mortality_rate
        self.symptoms_to_hospital_days = symptoms_to_hospital_days
        self.symptoms_to_mortality_days = symptoms_to_mortality_days

        # Hospitalization Parameters
        # https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf
        # Page 16
        self.hospitalization_rate_general = hospitalization_rate_general
        self.hospitalization_rate_icu = hospitalization_rate_icu
        self.hospitalization_length_of_stay_general = hospitalization_length_of_stay_general
        self.hospitalization_length_of_stay_icu = hospitalization_length_of_stay_icu
        self.hospitalization_length_of_stay_icu_and_ventilator = hospitalization_length_of_stay_icu_and_ventilator

        # beta as the transmission probability per contact times the rescale factor to rescale contact matrix data to
        # match expected R0
        self.beta = self._estimate_beta(self.R0)

        # http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf
        # = 0.53
        self.fraction_icu_requiring_ventilator = fraction_icu_requiring_ventilator

        # Capacity
        self.beds_general = beds_general
        self.beds_ICU = beds_ICU
        self.ventilators = ventilators

        self.mortality_rate_no_general_beds = mortality_rate_no_general_beds
        self.mortality_rate_no_ICU_beds = mortality_rate_no_ICU_beds
        self.mortality_rate_no_ventilator = mortality_rate_no_ventilator

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

    def calculate_R0(self, beta, S_fracs=None):
        """
        Using Next Generation Matrix method to calculate R0 given beta.

        Parameters
        ----------
        beta : float
            Transmission probability per contact times rescale factor that rescale contact matrix to match R0,
            can be greater than 1.

        Returns
        -------
        R0 : float
            Basic reproduction number.
        """

        # percentage of susceptible in each age group (assuming that initial condition is disease-free equilibrium)
        if S_fracs is None:
            S_fracs = self.N / self.N.sum()
        age_group_num = self.N.shape[0]
        # contact with susceptible at disease-free equilibrium
        # [C_11 * P_1, C_12 * P_1, ... C_1n * P_n]
        # ...
        # [C_n1 * P_n, C_n2 * P_n, ... C_nn * P_n]
        contact_with_susceptible = self.contact_matrix.values * S_fracs.T

        # transmission matrix with rates of immediate new infections into rows
        # due to transmission from columns
        T_E_E = np.zeros((age_group_num, age_group_num))  # from E to E
        T_E_A = contact_with_susceptible * beta  # A to E
        T_E_I = contact_with_susceptible * beta  # I to E
        T_A_E = np.zeros((age_group_num, age_group_num))  # E to A
        T_A_A = np.zeros((age_group_num, age_group_num))  # A to A
        T_A_I = np.zeros((age_group_num, age_group_num))  # I to A
        T_I_E = np.zeros((age_group_num, age_group_num))  # E to I
        T_I_A = np.zeros((age_group_num, age_group_num))  # A to I
        T_I_I = np.zeros((age_group_num, age_group_num))  # I to I
        T_E = np.concatenate([T_E_E, T_E_A, T_E_I], axis=1)  # all rates to E
        T_A = np.concatenate([T_A_E, T_A_A, T_A_I], axis=1)  # all rates to A
        T_I = np.concatenate([T_I_E, T_I_A, T_I_I], axis=1)  # all rates to I
        T = np.concatenate([T_E, T_A, T_I])

        # matrix of rates of transitions from rows to columns
        # rates of transition out of E (incubation or aging) and into E (aging)
        aging_rate_in = 1 / self.age_steps[:-1]
        aging_rate_out = 1 / self.age_steps
        Z_E_E = np.diag(-(aging_rate_out + self.sigma)) + np.diag(aging_rate_in, k=-1)
        Z_E_A = np.zeros((age_group_num, age_group_num))
        Z_E_I = np.zeros((age_group_num, age_group_num))

        # rates of transition out of A (recovery and aging) and into A (aging)
        Z_A_E = np.zeros((age_group_num, age_group_num))
        np.fill_diagonal(Z_A_E, self.sigma * (1 - self.gamma))   # transition from E to A
        Z_A_A = np.diag(-(aging_rate_out + self.delta)) + np.diag(aging_rate_in, k=-1)
        Z_A_I = np.zeros((age_group_num, age_group_num))

        # rates of transition out of A (recovery and aging) and into A (aging)
        Z_I_E = np.zeros((age_group_num, age_group_num))
        np.fill_diagonal(Z_I_E, self.sigma * self.gamma)         # transition from E to I
        Z_I_A = np.zeros((age_group_num, age_group_num))

        rate_infected_and_in_hospital_general = self.hospitalization_rate_general / self.symptoms_to_hospital_days
        rate_infected_and_in_hospital_icu = self.hospitalization_rate_icu / self.symptoms_to_hospital_days
        rate_infected_and_dead = self.mortality_rate / self.symptoms_to_mortality_days
        rate_out_of_I = aging_rate_out + self.delta + rate_infected_and_in_hospital_general + \
                        rate_infected_and_in_hospital_icu + rate_infected_and_dead
        Z_I_I = np.diag(-rate_out_of_I) + np.diag(aging_rate_in, k=-1)  # transition out of I
        Z_E = np.concatenate([Z_E_E, Z_E_A, Z_E_I], axis=1)
        Z_A = np.concatenate([Z_A_E, Z_A_A, Z_A_I], axis=1)
        Z_I = np.concatenate([Z_I_E, Z_I_A, Z_I_I], axis=1)
        Z = np.concatenate([Z_E, Z_A, Z_I])

        # Calculate R0 from transmission and transition matrix
        Z_inverse = np.linalg.inv(Z)
        K = T.dot(-Z_inverse)
        eigen_values = np.linalg.eigvals(K)
        R0 = max(eigen_values)  # R0 is the dominant eigenvalue

        return R0

    def _estimate_beta(self, expected_R0):
        """
        Estimate beta for given R0.

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
        R0s = list()
        betas = np.linspace(0, 10, 100)
        for beta in betas:
            R0 = self.calculate_R0(beta)
            R0s.append(R0)
        f = scipy.interpolate.interp1d(R0s, betas)
        beta = f(expected_R0)
        return float(beta.real)

    def calculate_Rt(self, S_fracs, suppression_policy):
        """
        Calculate R(t)
        """
        Rts = list()
        for n in range(S_fracs.shape[1]):
            Rt = self.calculate_R0(self.beta, S_fracs[:, n]) * suppression_policy[n]
            Rts.append(Rt)
        return Rts


    def _time_step(self, y, t):
        """
        One integral moment.

        y: array
            S, E, A, I, R, HNonICU, HICU, HICUVent, D = y
        """

        S, E, A, I, R, HNonICU, HICU, HICUVent, D = np.split(y[:-3], 9)

        # TODO: County-by-county affinity matrix terms can be used to describe
        # transmission network effects. ( also known as Multi-Region SEIR)
        # https://arxiv.org/pdf/2003.09875.pdf
        #  For those living in county i, the interacting county j exposure is given
        #  by A term dE_i/dt += N_i * Sum_j [ beta_j * mix_ij * I_j * S_i + beta_i *
        #  mix_ji * I_j * S_i ] mix_ij can be proxied by Census-based commuting
        #  matrices as workplace interactions are the dominant term. See:
        #  https://www.census.gov/topics/employment/commuting/guidance/flows.html
        #
        # TODO: Age-based contact mixing affinities.
        #    It is important to track demographics themselves as they impact
        #    hospitalization and mortality rates. Additionally, exposure rates vary
        #    by age, described by matrices linked below which need to be extracted
        #    from R for the US.
        #    https://cran.r-project.org/web/packages/socialmixr/vignettes/introduction.html
        #    For an infected age PMF vector I, and a contact matrix gamma dE_i/dT =
        #    S_i (*) gamma_ij I^j / N - gamma * E_i   # Someone should double check
        #    this

        # Effective contact rate * those that get exposed * those susceptible.
        total_ppl = (S + E + A + I + R).sum()

        # get matrix:
        # [C_11 * S_1 * I_1/N, ... C1j * S_1 * I_j/N...]
        # [C_21 * S_2 * I_1/N, ... C1j * S_2 * I_j/N...]
        # ...
        # [C_21 * S_n * I_1/N, ... C_21 * S_n * I_j/N ...]
        frac_infected = (self.kappa * I + A) / total_ppl
        S_and_I = S[:, np.newaxis].dot(frac_infected[:, np.newaxis].T)
        contacts_S_and_I = (S_and_I * self.contact_matrix).sum(axis=1)
        number_exposed = self.beta * self.suppression_policy(t) * contacts_S_and_I
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
        infected_and_in_hospital_general = I * self.hospitalization_rate_general / self.symptoms_to_hospital_days
        infected_and_in_hospital_icu = I * self.hospitalization_rate_icu / self.symptoms_to_hospital_days
        infected_and_dead = I * self.mortality_rate / self.symptoms_to_mortality_days

        age_in_I, age_out_I = self._aging_rate(I)
        dIdt = age_in_I + exposed_and_symptomatic - infected_and_recovered_no_hospital - \
               infected_and_in_hospital_general - infected_and_in_hospital_icu - infected_and_dead - age_out_I

        recovered_after_hospital_general = HNonICU / self.hospitalization_length_of_stay_general
        recovered_after_hospital_icu = HICU * ((1 - self.fraction_icu_requiring_ventilator)/ self.hospitalization_length_of_stay_icu
                                               + self.fraction_icu_requiring_ventilator / self.hospitalization_length_of_stay_icu_and_ventilator)

        age_in_HNonICU, age_out_HNonICU = self._aging_rate(HNonICU)
        dHNonICU_dt = age_in_HNonICU + infected_and_in_hospital_general - recovered_after_hospital_general - \
                      age_out_HNonICU

        age_in_HICU, age_out_HICU = self._aging_rate(HICU)
        dHICU_dt = age_in_HICU + infected_and_in_hospital_icu - recovered_after_hospital_icu - age_out_HICU

        # Tracking categories...
        dTotalInfections = sum(exposed_and_symptomatic) + sum(exposed_and_asymptomatic)
        dHAdmissions_general = sum(infected_and_in_hospital_general)
        dHAdmissions_ICU = sum(infected_and_in_hospital_icu)  # Ventilators also count as ICU beds.


        # This compartment is for tracking ventillator count. The beds are accounted for in the ICU cases.
        dHICUVent_dt = infected_and_in_hospital_icu * self.fraction_icu_requiring_ventilator \
                       - HICUVent / self.hospitalization_length_of_stay_icu_and_ventilator

        # Fraction that recover
        age_in_R, age_out_R = self._aging_rate(R)
        dRdt = (age_in_R
                + asymptomatic_and_recovered
                + infected_and_recovered_no_hospital
                + recovered_after_hospital_general
                + recovered_after_hospital_icu
                - age_out_R)

        # TODO Modify this based on increased mortality if beds saturated
        # TODO Age dep mortality. Recent estimate fo relative distribution Fig 3 here:
        #      http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf

        # Fraction that die.
        dDdt = infected_and_dead

        return np.concatenate([dSdt, dEdt, dAdt, dIdt, dRdt, dHNonICU_dt, dHICU_dt, dHICUVent_dt, dDdt,
                               np.array([dHAdmissions_general, dHAdmissions_ICU, dTotalInfections])])

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
        y0 = np.concatenate([self.S_initial, self.E_initial, self.A_initial, self.I_initial, self.R_initial,\
                             self.HGen_initial, self.HICU_initial, self.HICUVent_initial, self.D_initial,
                             np.array([HAdmissions_general, HAdmissions_ICU, TotalAllInfections])])

        # Integrate the SIR equations over the time grid, t.
        result_time_series = odeint(self._time_step, y0, self.t_list, atol=1e-3, rtol=1e-3)
        S, E, A, I, R, HGen, HICU, HICUVent, D = np.split(result_time_series.T[:-3], 9)
        HAdmissions_general, HAdmissions_ICU, TotalAllInfections = result_time_series.T[-3:]

        # derivatives to get e.g. deaths per day or admissions per day.
        total_hosp = np.array(HGen + HICU + HICUVent)

        S_fracs_within_age_group = S / S.sum(axis=0)

        Rt = self.calculate_Rt(S_fracs_within_age_group, self.suppression_policy(self.t_list))

        self.results = {
            't_list': self.t_list,
            'S': S.sum(axis=0),
            'E': E.sum(axis=0),
            'A': A.sum(axis=0),
            'I': I.sum(axis=0),
            'R': R.sum(axis=0),
            'HGen': HGen.sum(axis=0),
            'HICU': HICU.sum(axis=0),
            'HVent': HICUVent.sum(axis=0),
            'D': D.sum(axis=0),
            'Rt': Rt,
            'direct_deaths_per_day': np.array([0] + list(D.sum(axis=0)[1:] - D.sum(axis=0)[:-1])), # Derivative...
            # Here we assume that the number of person days above the saturation
            # divided by the mean length of stay approximates the number of
            # deaths from each source.
            # Ideally this is included in the dynamics, but this is left as a TODO.
            'deaths_from_hospital_bed_limits': np.cumsum((HGen - self.beds_general).clip(min=0)) *
                self.mortality_rate_no_general_beds / self.hospitalization_length_of_stay_general).sum(axis=0),
            # Here ICU = ICU + ICUVent, but we want to remove the ventilated fraction and account for that below.
            'deaths_from_icu_bed_limits': np.cumsum((HICU - self.beds_ICU).clip(min=0)) *
                self.mortality_rate_no_ICU_beds / self.hospitalization_length_of_stay_icu,
            #'deaths_from_ventilator_limits': np.cumsum((HICUVent - self.ventilators).clip(min=0)) *
            # self.mortality_rate_no_ventilator / self.hospitalization_length_of_stay_icu_and_ventilator,
            #'HGen_cumulative': np.cumsum(HGen) / self.hospitalization_length_of_stay_general,
            #'HICU_cumulative': np.cumsum(HICU) / self.hospitalization_length_of_stay_icu,
            #'HVent_cumulative': np.cumsum(HICUVent) / self.hospitalization_length_of_stay_icu_and_ventilator
        }

        # total_deaths = D + self.results['deaths_from_hospital_bed_limits'] \
        #                  + self.results['deaths_from_icu_bed_limits'] \
        #                  + self.results['deaths_from_ventilator_limits']
        #
        # self.results['total_new_infections'] = np.array([0] + list(TotalAllInfections[1:] - TotalAllInfections[:-1]))  # Derivative of the cumulative.
        # self.results['total_deaths_per_day'] = np.array([0] + list(total_deaths[1:] - total_deaths[:-1]))  # Derivative of the cumulative.
        # self.results['general_admissions_per_day'] = np.array([0] + list(HAdmissions_general[1:] - HAdmissions_general[:-1]))  # Derivative of the cumulative.
        # self.results['icu_admissions_per_day'] = np.array([0] + list(HAdmissions_ICU[1:] - HAdmissions_ICU[:-1]))  # Derivative of the cumulative.
        #
        # self.results['total_deaths'] =   self.results['deaths_from_hospital_bed_limits'] \
        #                                + self.results['deaths_from_icu_bed_limits'] \
        #                                + self.results['deaths_from_ventilator_limits'] \
        #                                + self.results['D']

    def plot_results(self, y_scale='log', by_age_group=False, xlim=None):
        """
        Generate a summary plot for the simulation.

        Parameters
        ----------
        y_scale: str
            Matplotlib scale to use on y-axis. Typically 'log' or 'linear'
        """
        if not by_age_group:
            # Plot the data on three separate curves for S(t), I(t) and R(t)
            fig = plt.figure(facecolor='w', figsize=(10, 8))
            plt.subplot(221)
            plt.plot(self.t_list, self.results['S'].sum(axis=0), alpha=1, lw=2, label='Susceptible')
            plt.plot(self.t_list, self.results['E'].sum(axis=0), alpha=.5, lw=2, label='Exposed')
            plt.plot(self.t_list, self.results['A'].sum(axis=0), alpha=.5, lw=2, label='Asymptomatic')
            plt.plot(self.t_list, self.results['I'].sum(axis=0), alpha=.5, lw=2, label='Infected')
            plt.plot(self.t_list, self.results['R'].sum(axis=0), alpha=1, lw=2, label='Recovered & Immune', linestyle='--')

            plt.plot(self.t_list, self.results['S'].sum(axis=0) + self.results['E'].sum(axis=0)
                     + self.results['A'].sum(axis=0) + self.results['I'].sum(axis=0)
                     + self.results['R'].sum(axis=0) + self.results['D'].sum(axis=0)
                     + self.results['HGen'].sum(axis=0) + self.results['HICU'].sum(axis=0),
                     label='Total')


            # This is debugging and should be constant.
            # TODO: we must be missing a small conservation term above.

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

            plt.plot(self.t_list, self.results['D'].sum(axis=0), alpha=.4, c='k', lw=1, label='Direct Deaths',
                     linestyle='-')
            # plt.plot(self.t_list, self.results['deaths_from_hospital_bed_limits'], alpha=1, c='k', lw=1, label='Deaths
            # From Bed Limits', linestyle=':')
            # plt.plot(self.t_list, self.results['deaths_from_icu_bed_limits'], alpha=1, c='k', lw=2, label='Deaths From ICU Bed Limits', linestyle='-.')
            # plt.plot(self.t_list, self.results['deaths_from_ventilator_limits'], alpha=1, c='k', lw=2, label='Deaths From No Ventillator', linestyle='--')
            # plt.plot(self.t_list, self.results['total_deaths'], alpha=1, c='k', lw=4, label='Total Deaths', linestyle='-')

            plt.plot(self.t_list, self.results['HGen'].sum(axis=0), alpha=1, lw=2, c='steelblue',
                     label='General Beds Required', linestyle='-')
            plt.hlines(self.beds_general, self.t_list[0], self.t_list[-1], 'steelblue', alpha=1, lw=2, label='ICU Bed Capacity', linestyle='--')

            plt.plot(self.t_list, self.results['HICU'].sum(axis=0), alpha=1, lw=2, c='firebrick', label='ICU Beds Required', linestyle='-')
            plt.hlines(self.beds_ICU, self.t_list[0], self.t_list[-1], 'firebrick', alpha=1, lw=2, label='General Bed Capacity', linestyle='--')

            plt.plot(self.t_list, self.results['HVent'].sum(axis=0), alpha=1, lw=2, c='seagreen', label='Ventilators Required', linestyle='-')
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

            # Reproduction numbers
            plt.subplot(223)
            plt.plot(self.t_list, [self.suppression_policy(t) for t in self.t_list], c='steelblue')
            plt.ylabel('Contact Rate Reduction')
            plt.xlabel('Time [days]', fontsize=12)
            plt.grid(True, which='both')

            plt.subplot(224)
            plt.plot(self.t_list, self.results['Rt'], c='steelblue')
            plt.ylabel('R(t)')
            plt.xlabel('Time [days]', fontsize=12)
            plt.grid(True, which='both')

            plt.tight_layout()

        else:
            # Plot the data by age group
            fig, axes = plt.subplots(len(self.age_groups), 2, figsize=(10, 50))
            for n, age_group in enumerate(self.age_groups):
                ax1, ax2 = axes[n]
                ax1.plot(self.t_list, self.results['S'][n, :], alpha=1, lw=2, label='Susceptible')
                ax1.plot(self.t_list, self.results['E'][n, :], alpha=.5, lw=2, label='Exposed')
                ax1.plot(self.t_list, self.results['A'][n, :], alpha=.5, lw=2, label='Asymptomatic')
                ax1.plot(self.t_list, self.results['I'][n, :], alpha=.5, lw=2, label='Infected')
                ax1.plot(self.t_list, self.results['R'][n, :], alpha=1, lw=2, label='Recovered & Immune',
                         linestyle='--')
                ax2.plot(self.t_list, self.results['HGen'][n, :], alpha=1, lw=2, label='Hospital general',
                         linestyle='--')
                ax2.plot(self.t_list, self.results['HICU'][n, :], alpha=1, lw=2, label='ICU',
                         linestyle='--')
                ax2.plot(self.t_list, self.results['D'][n, :], alpha=1, lw=2, label='direct death',
                         linestyle='--')
                ax1.legend()
                ax2.legend()
                ax1.set_xlabel('days')
                ax2.set_xlabel('days')
                ax1.set_title('age group %d-%d' %(age_group[0], age_group[1]))
                ax1.set_yscale('log')
                ax2.set_yscale('log')
                ax1.set_ylim(ymin=1)
                ax2.set_ylim(ymin=1)

            plt.tight_layout()


        #return fig
