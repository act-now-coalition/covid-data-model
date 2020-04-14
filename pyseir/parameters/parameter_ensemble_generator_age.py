import numpy as np
import pandas as pd
import us
from pyseir import load_data
from libs.datasets import FIPSPopulation
from libs.datasets import DHBeds
from libs.datasets.dataset_utils import AggregationLevel
import scipy

beds_data = None
population_data = None
hosp_data = None

class ParameterEnsembleGeneratorAge:
    """
    Generate ensembles of parameters for SEIR modeling with age structure.

    Parameters
    ----------
    fips: str
        County or state fips code.
    N_samples: int
        Integer number of samples to generate.
    t_list: array-like
        Array of times to integrate against.
    I_initial: int
        Initial infected case count to consider.
    suppression_policy: callable(t): pyseir.model.suppression_policy
        Suppression policy to apply.
    """
    def __init__(self, fips, N_samples, t_list,
                 I_initial=1, suppression_policy=None):

        # Caching globally to avoid relatively significant performance overhead
        # of loading for each county.
        global beds_data, population_data, hosp_data
        if (not beds_data) or (not population_data):
            beds_data = DHBeds.local().beds()
            population_data = FIPSPopulation.local().population()
        if hosp_data is None:
            hosp_data = load_data.load_cdc_hospitalization_data()

        self.fips = fips
        self.agg_level = AggregationLevel.COUNTY if len(self.fips) == 5 else AggregationLevel.STATE
        self.N_samples = N_samples
        self.I_initial = I_initial
        self.suppression_policy = suppression_policy
        self.t_list = t_list
        self.contact_matrix_data = load_data.load_contact_matrix_data_by_fips(self.fips)
        self.population = np.array(self.contact_matrix_data[self.fips]['age_distribution'])

        if self.agg_level is AggregationLevel.COUNTY:
            self.county_metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
            self.state_abbr = us.states.lookup(self.county_metadata['state']).abbr
            # TODO: Some counties do not have hospitals. Likely need to go to HRR level..
            self.beds = beds_data.get_county_level(self.state_abbr, fips=self.fips) or 0
            self.icu_beds = beds_data.get_county_level(self.state_abbr, fips=self.fips, column='icu_beds') or 0
        else:
            self.state_abbr = us.states.lookup(fips).abbr
            self.beds = beds_data.get_state_level(self.state_abbr) or 0
            self.icu_beds = beds_data.get_state_level(self.state_abbr, column='icu_beds') or 0

    def generate_age_specific_rates(self):
        """
        Generate age specific hospitalization_rate_general,
        hospitalization_rate_icu, and mortality_rate.
        """
        age_bin_edges = self.contact_matrix_data[self.fips]['age_bin_edges']
        age_bin_centers = (np.array(age_bin_edges[1:]) + np.array(age_bin_edges[:-1])) / 2

        for suffix in ['_hgen', '_icu', '_fatility']:
            f = scipy.interpolate.interp1d(
                hosp_data['lower_age'].tolist() + hosp_data['mean_age'].tolist(),
                hosp_data['lower%s' % suffix].tolist() + hosp_data['mean%s' % suffix].tolist())
            yield f(age_bin_centers).clip(min=0)


    def generate_age_specific_initial_conditions(self):
        """
        Generate initial condition based on age distribution.

        Returns
        -------

        """
        age_dist = self.contact_matrix_data[self.fips]['age_distribution']
        E_initial = np.zeros(len(age_dist))
        A_initial = np.zeros(len(age_dist))
        I_initial = self.I_initial * np.array(age_dist) / sum(age_dist)
        HGen_initial = np.zeros(len(age_dist))
        HICU_initial = np.zeros(len(age_dist))
        HICUVent_initial = np.zeros(len(age_dist))

        return E_initial, A_initial, I_initial, HGen_initial, HICU_initial, HICUVent_initial

    def sample_seir_parameters(self, override_params=None):
        """
        Generate N_samples of parameter values from the priors listed below.

        Parameters
        ----------
        override_params: dict()
            Individual parameters can be overridden here.

        Returns
        -------
        : list(dict)
            List of parameter sets to feed to the simulations.
        """
        override_params = override_params or dict()
        parameter_sets = []
        E_initial, A_initial, I_initial, HGen_initial, HICU_initial, HICUVent_initial = \
            self.generate_age_specific_initial_conditions()

        hospitalization_rate_general, hospitalization_rate_icu, mortality_rate = \
            self.generate_age_specific_rates()

        # shift to have mean 0.4
        mortality_rate_from_ICU = \
            mortality_rate + 0.4 - mortality_rate.mean()

        contact_matrix = np.array(self.contact_matrix_data[self.fips]['contact_matrix'])
        age_bin_edges = np.array(self.contact_matrix_data[self.fips]['age_bin_edges'])

        for _ in range(self.N_samples):

            # For now we have disabled this bucket and lowered rates of other
            # boxes accordingly. Since we were not modeling different contact
            # rates, this has the same result.
            fraction_asymptomatic = 0

            parameter_sets.append(dict(
                t_list=self.t_list,
                N=self.population,
                A_initial=A_initial,
                I_initial=I_initial,
                R_initial=0,
                E_initial=E_initial,
                D_initial=0,
                HGen_initial=HGen_initial,
                HICU_initial=HICU_initial,
                HICUVent_initial=HICUVent_initial,
                suppression_policy=self.suppression_policy,
                age_bin_edges=age_bin_edges,
                contact_matrix=np.random.normal(loc=contact_matrix, scale=contact_matrix/10).clip(min=0),
                R0=np.random.uniform(low=3.2, high=4),
                R0_hospital=np.random.uniform(low=3.2 / 6, high=4 / 6),
                # These parameters produce an IFR ~0.0065 if we had infinite
                # capacity, and about ~0.0125 with capacity constraints imposed
                hospitalization_rate_general=np.random.normal(loc=hospitalization_rate_general,
                                                              scale=hospitalization_rate_general/10).clip(min=0),
                hospitalization_rate_icu=np.random.normal(loc=hospitalization_rate_icu,
                                                          scale=hospitalization_rate_icu/10).clip(min=0),
                fraction_icu_requiring_ventilator=max(np.random.normal(loc=0.6, scale=0.1), 0),
                sigma=1 / np.random.normal(loc=3., scale=0.86),  # Imperial college - 2 days since that is expected infectious period.
                delta=1 / np.random.gamma(6.0, scale=1),  # Kind of based on imperial college + CDC digest.
                delta_hospital=1 / np.random.gamma(8.0, scale=1),  # Kind of based on imperial college + CDC digest.
                kappa=1, # Contact rate for asympt
                gamma=(1-fraction_asymptomatic),
                # https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html
                symptoms_to_hospital_days=np.random.normal(loc=6., scale=1.5),
                symptoms_to_mortality_days=np.random.normal(loc=18.8, scale=.45), # Imperial College
                hospitalization_length_of_stay_general=np.random.normal(loc=6, scale=1),
                hospitalization_length_of_stay_icu=np.random.normal(loc=14, scale=3),
                hospitalization_length_of_stay_icu_and_ventilator=np.random.normal(loc=15, scale=3),
                # if you assume the ARDS population is the group that would die
                # w/o ventilation, this would suggest a 20-42% mortality rate
                # among general hospitalized patients w/o access to ventilators:
                # “Among all patients, a range of 3% to 17% developed ARDS
                # compared to a range of 20% to 42% for hospitalized patients
                # and 67% to 85% for patients admitted to the ICU.1,4-6,8,11”

                # 10% Of the population should die at saturation levels. CFR
                # from Italy is 11.9% right now, Spain 8.9%.  System has to
                # produce,
                mortality_rate_no_general_beds=np.random.normal(loc=.05, scale=0.01),
                mortality_rate_from_hospital=0,
                mortality_rate_from_ICU=np.random.normal(loc=mortality_rate_from_ICU,
                                                         scale=mortality_rate_from_ICU/10).clip(min=0),
                mortality_rate_from_ICUVent=0.60,
                mortality_rate_no_ICU_beds=1.0,
                beds_general=self.beds * 0.4 * 2.07, # 60% utliization, no scaling...
                # TODO.. Patch this After Issue 132
                beds_ICU= (1 - 0.85) * self.icu_beds,  # No scaling, 85% utilization...
                # hospital_capacity_change_daily_rate=1.05,
                # max_hospital_capacity_factor=2.07,
                # initial_hospital_bed_utilization=0.6,
                # Rubinson L, Vaughn F, Nelson S, et al. Mechanical ventilators
                # in US acute care hospitals. Disaster Med Public Health Prep.
                # 2010;4(3):199-206. http://dx.doi.org/10.1001/dmp.2010.18.
                # 0.7 ventilators per ICU bed on average in US ~80k Assume
                # another 20-40% of 100k old ventilators can be used. = 100-120
                # for 100k ICU beds
                # TODO: Update this if possible by county or state. The ref above has state estimates
                # Staff expertise may be a limiting factor:
                # https://sccm.org/getattachment/About-SCCM/Media-Relations/Final-Covid19-Press-Release.pdf?lang=en-US
                # TODO: Patch after #133
                ventilators=self.icu_beds * np.random.uniform(low=1.0, high=1.2),
            ))

        for parameter_set in parameter_sets:
            parameter_set.update(override_params)

        return parameter_sets

    def get_average_seir_parameters(self):
        """
        Sample from the ensemble to obtain the average parameter values.

        Returns
        -------
        average_parameters: dict
            Average of the parameter ensemble, determined by sampling.
        """
        df = pd.DataFrame(self.sample_seir_parameters()).drop(['t_list', 'suppression_policy'], axis=1)
        average_parameters = {}
        for col in df.columns:
            average_parameters[col] = df[col].mean()
        average_parameters['t_list'] = self.t_list
        average_parameters['suppression_policy'] = self.suppression_policy
        return average_parameters
