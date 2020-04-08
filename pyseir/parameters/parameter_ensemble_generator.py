import numpy as np
import pandas as pd
from pyseir import load_data
from libs.datasets import FIPSPopulation
from libs.datasets import DHBeds
import us


beds_data = None
population_data = None


class ParameterEnsembleGenerator:
    """
    Generate ensembles of parameters for SEIR modeling.

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
        global beds_data, population_data
        if not beds_data or not population_data:
            beds_data = DHBeds.local().beds()
            population_data = FIPSPopulation.local().population()

        self.fips = fips
        self.geographic_unit = 'county' if len(self.fips) == 5 else 'state'
        self.N_samples = N_samples
        self.I_initial = I_initial
        self.suppression_policy = suppression_policy
        self.t_list = t_list

        if self.geographic_unit == 'county':
            self.county_metadata = load_data.load_county_metadata().set_index('fips').loc[fips].to_dict()
            self.state_abbr = us.states.lookup(self.county_metadata['state']).abbr
            self.population = population_data.get_county_level('USA', state=self.state_abbr, fips=self.fips)
            # TODO: Some counties do not have hospitals. Likely need to go to HRR level..
            self.beds = beds_data.get_county_level(self.state_abbr, fips=self.fips) or 0
        else:
            self.state_abbr = us.states.lookup(fips).abbr
            self.population = population_data.get_state_level('USA', state=self.state_abbr)
            self.beds = beds_data.get_state_level(self.state_abbr) or 0

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
        for _ in range(self.N_samples):

            # https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html
            # TODO: 10% is being used by CA group.  CDC suggests 20% case hospitalization rate
            # Note that this is 10% of symptomatic cases, making overall hospitalization around 5%.
            # https: // www.statista.com / statistics / 1105402 / covid - hospitalization - rates - us - by - age - group /
            hospitalization_rate_general = np.random.normal(loc=0.125, scale=0.03)
            fraction_asymptomatic = np.random.uniform(0.4, 0.6)

            parameter_sets.append(dict(
                t_list=self.t_list,
                N=self.population,
                A_initial=fraction_asymptomatic * self.I_initial / (1 - fraction_asymptomatic), # assume no asymptomatic cases are tested.
                I_initial=self.I_initial,
                R_initial=0,
                E_initial=0,
                D_initial=0,
                HGen_initial=0,
                HICU_initial=0,
                HICUVent_initial=0,
                suppression_policy=self.suppression_policy,
                R0=np.random.uniform(low=3, high=4.5),            # Imperial College
                R0_hospital=np.random.uniform(low=.5, high=4.5 / 6),  # Imperial College
                hospitalization_rate_general=hospitalization_rate_general,
                # https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html
                hospitalization_rate_icu=max(np.random.normal(loc=.29, scale=0.03) * hospitalization_rate_general, 0),
                # http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf
                # Coronatracking.com/data
                fraction_icu_requiring_ventilator=max(np.random.normal(loc=0.44, scale=0.1), 0),
                sigma=1 / np.random.normal(loc=3.1, scale=0.86),  # Imperial college - 2 days since that is expected infectious period.
                delta=1 / np.random.gamma(6.0, scale=1),  # Kind of based on imperial college + CDC digest.
                delta_hospital=1 / np.random.gamma(8.0, scale=1),  # Kind of based on imperial college + CDC digest.
                kappa=1,
                gamma=fraction_asymptomatic,
                # https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html
                symptoms_to_hospital_days=np.random.normal(loc=6.5, scale=1.5),
                symptoms_to_mortality_days=np.random.normal(loc=18.8, scale=.45), # Imperial College
                    hospitalization_length_of_stay_general=np.random.normal(loc=7, scale=2),
                    hospitalization_length_of_stay_icu=np.random.normal(loc=16, scale=3),
                    hospitalization_length_of_stay_icu_and_ventilator=np.random.normal(loc=17, scale=3),
                mortality_rate=np.random.normal(loc=0.0109, scale=0.0025),
                # if you assume the ARDS population is the group that would die
                # w/o ventilation, this would suggest a 20-42% mortality rate
                # among general hospitalized patients w/o access to ventilators:
                # “Among all patients, a range of 3% to 17% developed ARDS
                # compared to a range of 20% to 42% for hospitalized patients
                # and 67% to 85% for patients admitted to the ICU.1,4-6,8,11”

                # 10% Of the population should die at saturation levels. CFR
                # from Italy is 11.9% right now, Spain 8.9%.  System has to
                # produce,
                mortality_rate_no_general_beds=np.random.normal(loc=.12, scale=0.04),
                # Bumped these up a bit. Dyspnea -> ARDS -> Septic Shock all
                # very fatal.
                mortality_rate_no_ICU_beds=np.random.uniform(low=0.8, high=1),
                mortality_rate_no_ventilator=1,
                # beds_general=  self.county_metadata_merged.get('num_staffed_beds', 0)
                #              - self.county_metadata_merged.get('bed_utilization', 0),
                #              # + self.county_metadata_merged.get('potential_increase_in_bed_capac', 0),
                beds_general=self.beds * 0.4 * 2.07,
                # TODO.. Patch this After Issue 132
                beds_ICU=0, # self.county_metadata_merged.get('num_icu_beds', 0),
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
                ventilators=0 #self.county_metadata_merged.get('num_icu_beds', 0) * np.random.uniform(low=1.0, high=1.2)
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
        df = pd.DataFrame(self.sample_seir_parameters()).drop('t_list', axis=1)
        average_parameters = df.mean().to_dict()
        average_parameters['t_list'] = self.t_list
        return average_parameters
