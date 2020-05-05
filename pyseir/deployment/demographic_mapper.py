import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from enum import Enum
from datetime import datetime, timedelta


class CovidMeasure(Enum):
    """
    - IHR: probability an infected person is hospitalized, including non-ICU
           and ICU admission.
    - IHR_icu: probability an infected person is admitted to ICU.
    - IHR_general: probability an infected person is admitted to non-ICU.
    - HR: probability a person gets hospitalized due to covid-19, including
          co-occurrence of infection and admission to hospital.
    - HR_icu: probability a person gets admitted to ICU due to covid-19,
              including co-occurrence of infection and admission to ICU.
    - HR_general: probability a person gets admitted to non-ICU due to
                  covid-19, including co-occurrence of infection and
                  admission to non-ICU.
    - IFR: probability an infected person dies of covid-19.
    """

    IHR_GENERAL = 'IHR_general'
    IHR_ICU = 'IHR_icu'
    IHR = 'IHR'
    HR_GENERAL = 'HR_general'
    HR_ICU = 'HR_icu'
    HR = 'HR'
    IFR = 'IFR'


class CovidMeasureUnit(Enum):
    PER_CAPITA = 'per_capita'
    PER_CAPITA_DAY = 'per_capita_day'


class DemographicMapper:
    """
    Maps SEIR model inference to a target population based on the target
    population's demographic distribution. Currently supports mapping based
    on age structure.

    The mapper calculates:
    i. size of population at each state of infection (susceptible,
    exposed, infected etc.) which are predicted by the MLE model and mapped
    to target demographic distribution (currently supports age groups);
    ii. probabilities of hospitalization (IHR) or death (IFR) by given 
    demographic category, depending on the measure.
    For detailed description of measure's meanings, check CovidMeasure.

    The measure may quantify the probability of a future outcome or a
    daily event, depending on the measure_unit: when measure unit is `per
    capita`, the measure quantifies the probability that a future event will
    ultimately occur; when measure unit is `per capita day`, the measure
    quantifies the probability of an event per day.

    The final results include:
    i. time series of population size at different states of infection
       summed over demographic groups and mapped to the target demographic
       distribution.
    ii. time series of measures averaged through demographic groups weighted
        by the target demographic distribution. If risk_modifier_by_age is
        specified, it will be used as relative risk of target population
        compared to general population risk per age group to further modify
        the weights.

    Attributes
    ----------
    predictions: dict
        Contains time series of SEIR model compartment size summed
        over age groups. Time series are simulated with MLE parameters.
        With keys:
        - 'E': np.array, exposed
        - 'I': np.array, infected and symptomatic
        - 'A': np.array, infected and asymptomatic
        - 'HGen': np.array, admitted to non-ICU
        - 'HICU': np.array, admitted to ICU
        - 'HVent': np.array, admitted to ICU with ventilator
        - 'D': np.array, death during hospitalization
        - 'R': recovered
    predictions_by_age: dict
        Contains time series of SEIR model compartment size by age groups.
        Time series are simulated with MLE parameters.
        With keys:
        - 'E': np.array, exposed by age groups
        - 'I': np.array, infected and symptomatic by age groups
        - 'A': np.array, infected and asymptomatic by age groups
        - 'HGen': np.array, admitted to non-ICU by age groups
        - 'HICU': np.array, admitted to ICU by age groups
        - 'HVent': np.array, admitted to ICU with ventilator by age groups
    parameters: dict
        Contains MLE model parameters. For full list of parameters,
        refer description of parameters of SEIRModelAge.
    measures: list(CovidMeasure)
        Covid measures.
    measure_units: list(CovidMeasureUnit)
        Units of covid measure.
    IHR: dict
        Rates of hospitalization by age group, type of hospitalization,
        and unit of rates, with type of hospitalization as primary key,
        unit as secondary key, and array of corresponding time series of
        rates as values.
        For example, IHR['HGen']['per_capita'] is the
        time series of probability of being admitted to non-ICU per
        capita among infected population (asymptomatic + symptomatic).
    IFR: dict
        Rates of mortality by age group, type of hospitalization,
        and unit of rates, with type of hospitalization as primary key,
        unit as secondary key, and array of corresponding time series of
        rates as values.
        For example, IFR['HICU']['per_capita'] is the time
        series of probability of death in ICU per capita among infected
        population (asymptomatic + symptomatic + hospitalized).
    prevalence: np.array
        Age-specific prevalence time series simulated with SEIR model
        with MLE parameters.
    results: dict
        Contains:
            - compartments:
              - <compartment>: time series of population at a specific
                               infection state (susceptible, infected,
                               hospitalized, etc.) simulated by MLE model
                               assuming the population has the demographic
                               distribution of the target population. Each time
                               series is recorded as pd.DataFrame, with dates of
                               prediction as index.
              name of compartment include: S - susceptible, E - exposed,
              A - asymptomatic, I - symptomatic, HGen - in non-ICU, HICU -
              in ICU, HVent - on ventilator, N - entire population.
            - <measure>:
              - <measure_unit>: time series of covid measures predicted using
                                the MLE model and averaged over target
                                demographic distribution (adjusted by risk
                                modification if relative risk is specified).
                                The time series is recorded as pd.DataFrame,
                                with dates of prediction as index.



    Parameters
    ----------
    fips: str
        State or county FIPS code
    mle_model: SEIRModelAge
        Model with age structure and MLE model parameters
    fit_results: dict
        MLE epi parameters and associated errors. The parameter used is
        t0_date.
    measures: str or list(str)
        Names of covid measures, should be interpretable by CovidMeasure.
    measure_units: str or list(str)
        Units of covid measures, should be interpretable by CovidMeasureUnit.
    target_age_distribution_pdf: callable
        Function that takes an array of age as input returns PDF of the age
        based on the age distribution of target population.
    risk_modifier_by_age: dict(callable)
        Contains:
        - <measure>: Function that returns risk ratios by age group that
                     modifies the risk of corresponding measure (
                     hospitalization or mortality rate).
        measure should be interpretable by CovidMeasure.
    """

    def __init__(self,
                 fips,
                 mle_model,
                 fit_results,
                 measures=None,
                 measure_units=None,
                 target_age_distribution_pdf=None,
                 risk_modifier_by_age=None):

        self.fips = fips
        self.predictions = {k: v for k, v in mle_model.results.items() if k != 'by_age'}
        self.predictions_by_age = mle_model.results['by_age']
        self.parameters = {k: v for k, v in mle_model.__dict__.items() if k not in ('by_age', 'results')}
        self.fit_results = fit_results

        if measures is not None:
            measures = [measures] if not isinstance(measures, list) else measures
            measures = [CovidMeasure(m) for m in measures]
        self.measures = measures

        if measure_units is not None:
            measure_units = [measure_units] if not isinstance(measure_units, list) else measure_units
            measure_units = [CovidMeasureUnit(u) for u in measure_units]
        self.measure_units = measure_units

        if target_age_distribution_pdf is None:
            target_age_distribution_pdf = lambda x: np.ones(len(self.parameters['age_groups']))
        self.target_age_distribution_pdf = target_age_distribution_pdf
        self.risk_modifier_by_age = risk_modifier_by_age

        # get parameters required to calculate covid measures
        self.IHR, self.IFR = self._generate_IHR_IFR()
        self.prevalence = self._age_specific_prevalence()

        self.results = None


    def _age_specific_prevalence(self):
        """
        Calculate age specific prevalence.

        Returns
        -------
        prevalence: np.array
            Trajectory of covid infection prevalence inferred by the MLE
            model.
        """
        prevalence = (self.predictions_by_age['I']
                    + self.predictions_by_age['A'])/ self.parameters['N'][:, np.newaxis]
        return prevalence

    def _reconstruct_mortality_inflow_rates(self):
        """
        Reconstruct trajectory of inferred per-capita rates of inflow
        of mortality through time from MLE model parameters and compartments.

        Returns
        -------
        mortality_inflow_rates: dict
            Contains:
            - <hospitalization>: np.array
              Age-specific per capita mortality rate in given type of hospitalization.
            hospitalization include: HGen, HICU, HVent.
        """

        # per capit mortality rate from people who need ICU or in ICU
        mortality_rate_ICU = np.tile(self.parameters['mortality_rate_from_ICU'],
                                     (self.parameters['t_list'].shape[0], 1)).T
        idx_inadequate_icu_bed = np.where(self.predictions['HICU'] > self.parameters['beds_ICU'])
        frac_no_access_to_icu = (self.predictions['HICU'] - self.parameters['beds_ICU']) / self.predictions['HICU']
        frac_no_access_to_icu = frac_no_access_to_icu[idx_inadequate_icu_bed]
        if len(frac_no_access_to_icu) > 0:
            mortality_rate_ICU[:, idx_inadequate_icu_bed] = \
                np.tile(self.parameters['mortality_rate_from_ICU'] * (1 - frac_no_access_to_icu[:, np.newaxis])
                      + self.parameters['mortality_rate_no_ICU_beds'] * frac_no_access_to_icu,
                        (1, mortality_rate_ICU.shape[0]))

        # per capita mortality rate from people who need to be hospitalized or in hospital
        mortality_rate_NonICU = np.tile(self.parameters['mortality_rate_from_hospital'],
                                        (self.parameters['t_list'].shape[0], 1)).T
        idx_inadequate_hgen_bed = np.where(self.predictions['HGen'] > self.parameters['beds_general'])
        frac_no_access_to_hgen = (self.predictions['HGen'] - self.parameters['beds_general']) / self.predictions['HGen']
        frac_no_access_to_hgen = frac_no_access_to_hgen[idx_inadequate_hgen_bed]

        if len(frac_no_access_to_hgen) > 0:
            mortality_rate_NonICU[:, idx_inadequate_hgen_bed] = \
                np.tile(self.parameters['mortality_rate_from_hospital'] * (1 - frac_no_access_to_hgen)
                      + self.parameters['mortality_rate_no_general_beds'] * frac_no_access_to_hgen,
                        (1, mortality_rate_NonICU.shape[0]))

        mortality_inflow_rates = {}
        mortality_inflow_rates['HGen'] = \
            mortality_rate_NonICU / self.parameters['hospitalization_length_of_stay_general']
        mortality_inflow_rates['HICU'] = \
            (1 - self.parameters['fraction_icu_requiring_ventilator']) * mortality_rate_ICU / \
                             self.parameters['hospitalization_length_of_stay_icu']
        mortality_inflow_rates['HVent'] = self.parameters['mortality_rate_from_ICUVent'] / \
                                  self.parameters['hospitalization_length_of_stay_icu_and_ventilator']

        return mortality_inflow_rates

    def _reconstruct_hospitalization_inflow_rates(self):
        """
        Reconstruct trajectory of inferred per capita rates of inflow to
        hospitalization through time from MLE model parameters and
        compartments.

        Returns
        -------
        hospitalization_inflow_rates: dict
            Contains:
            - <hospitalization>: : np.array
              Age specific per-capita rate of admission to corresponding
              hospitalization.
            hospitalization include: HGen, HICU, HVent.
        """
        hospitalization_inflow_rates = {}
        hospitalization_inflow_rates['HGen'] = \
            (self.parameters['hospitalization_rate_general']
           - self.parameters['hospitalization_rate_icu']) / self.parameters['symptoms_to_hospital_days']
        hospitalization_inflow_rates['HICU'] = self.parameters['hospitalization_rate_icu'] / self.parameters[
            'symptoms_to_hospital_days']
        hospitalization_inflow_rates['HVent'] = \
            hospitalization_inflow_rates['HICU'] * self.parameters['fraction_icu_requiring_ventilator']

        return hospitalization_inflow_rates

    def _reconstruct_hospital_recovery_inflow_rates(self):
        """
        Reconstruct trajectory of inferred per capita rates of recovery
        during hospitalization through time from MLE model parameters and
        compartments.

        Returns
        -------
        hospital_recovery_inflow_rates: dict
            Contains:
            - <hospitalization>: np.array, age-specific rate of recovery
                                 from the corresponding hospitalization.
              hospitalization include: HGen, HICU, HVent.
        """
        mortality_inflow_rates = self._reconstruct_mortality_inflow_rates()

        hospital_recovery_inflow_rates = {}
        hospital_recovery_inflow_rates['HGen'] = (1 - mortality_inflow_rates['HGen']) / \
            self.parameters['hospitalization_length_of_stay_general']
        hospital_recovery_inflow_rates['HICU'] = (1 - mortality_inflow_rates['HICU']) * (1 - self.parameters[
            'fraction_icu_requiring_ventilator']) \
            / self.parameters['hospitalization_length_of_stay_icu']
        hospital_recovery_inflow_rates['HVent'] = \
            (1 - np.maximum(mortality_inflow_rates['HVent'],
                            self.parameters['mortality_rate_from_ICUVent'])) \
            / self.parameters['hospitalization_length_of_stay_icu_and_ventilator']

        return hospital_recovery_inflow_rates

    def _age_specific_IHR_by_hospitalization(self, measure_unit):
        """
        Calculate age specific hospitalization rates among infected
        population (IHR) for a given measure unit by types of
        hospitalization.

        When unit is per capita, calculates the probability that an infected
        person (asymptomatic or symptomatic) ultimately gets admitted to
        hospital:
            probability of being symptomatic * rate of hospitalized / (rate of
            hospitalized + rate of recovery without hospitalization)

        When unit is per capita day, calculate the probability that an
        infected person gets admitted to hospital per day:
           new hospitalization / (asymptomatic + symptomatic infections)

        Parameters
        ----------
        measure_unit: CovidMeasureUnit
            Unit of covid measure

        Returns
        -------
        IHR: dict
            Contains:
            - <hospitalization>: dict
              - <measure_unit>:  np.array
                Rates of admission to hospital of infected population
                by types of hospitalization.
              hospitalization include: HGen, HICU, HVent.
        """

        hospital_inflow_rates = self._reconstruct_hospitalization_inflow_rates()

        fraction_of_symptomatic = self.predictions_by_age['I'] / (self.predictions_by_age['I']
                                                                  + self.predictions_by_age['A'])
        IHR = defaultdict(dict)
        if measure_unit is CovidMeasureUnit.PER_CAPITA:
            HR = hospital_inflow_rates

            for key in HR:
                IHR[key][measure_unit.value] = HR[key][:, np.newaxis] * fraction_of_symptomatic

        elif measure_unit is CovidMeasureUnit.PER_CAPITA_DAY:
            total_rate_out_of_I = self.parameters['delta']
            for key in hospital_inflow_rates:
                total_rate_out_of_I += hospital_inflow_rates[key]

            HR = {}
            for key in hospital_inflow_rates:
                HR[key] = np.tile(hospital_inflow_rates[key] / total_rate_out_of_I,
                                  (len(self.parameters['t_list']), 1)).T
            for key in HR:
                IHR[key][measure_unit.value] = HR[key] * fraction_of_symptomatic

        return IHR

    def _age_specific_IFR_by_hospitalization(self, measure_unit, IHR):
        """
        Calculates age specific mortality with given unit among infected
        population (IFR) by types of hospitalization.

        When unit is per capita, mortality rate is calculated as the
        probability that an infected person dies of covid-19:
            probability of hospitalization * mortality rate /
            (mortality rate + recovery rate)

        When unit is per capita day, mortality rate is calculated as
        probability of dying of covid-19 per capita per day among infected
        population:
            new death / (asymptomatic + symptomatic + hospitalized)

        Parameters
        ----------
        measure_unit: CovidMeasureUnit
            Unit of the covid measure.
        IHR: dict
            Rates of hospitalization by age group, type of hospitalization,
            and unit of rates, with type of hospitalization as primary key,
            unit as secondary key, and array of corresponding time series of
            rates as values.
            For example, IHR['HGen']['per_capita'] is the
            time series of probability of being admitted to non-ICU per
            capita among infected population (asymptomatic + symptomatic).

        Returns
        -------
        IFR_general: np.array
            Rates of mortality in non-ICU of infected population
        IFR_icu: np.array
            Rates of mortality in ICU of infected population
        IFR_icu_vent: np.array
            Rates of mortality in ICU with Ventilator of infected population
        """
        mortality_inflow_rates = self._reconstruct_mortality_inflow_rates()

        hospital_recovery_inflow_rates = self._reconstruct_hospital_recovery_inflow_rates()

        mortality_probs = {}
        for key in hospital_recovery_inflow_rates:
            mortality_probs[key] = \
                mortality_inflow_rates[key] / (mortality_inflow_rates[key] + hospital_recovery_inflow_rates[key])

        IFR = defaultdict(dict)

        for key in mortality_probs:
            IFR[key][measure_unit.value] = mortality_probs[key] * IHR[key][measure_unit.value]

        if measure_unit is CovidMeasureUnit.PER_CAPITA_DAY:
            total_infections = np.zeros(self.predictions_by_age['I'].shape)
            for c in ['I', 'A', 'HGen', 'HICU', 'HVent']:
                total_infections += self.predictions_by_age[c]

            for key in mortality_inflow_rates:
                IFR[key][measure_unit.value] = \
                    mortality_inflow_rates[key] * self.predictions_by_age[key] / total_infections

        return IFR

    def _generate_IHR_IFR(self):
        """
        Generates age specific hospitalization rates and mortality rates
        among infected population for given unit (per capita or per capita day).

        Returns
        -------
        IHR: dict
            Rates of hospitalization by age group, type of hospitalization,
            and unit of rates, with unit as primary key, type of
            hospitalization as secondary key, and array of corresponding
            time series of rates as values.
            For example, IHR['HGen']['per_capita'] is the
            time series of probability of being admitted to non-ICU per
            capita among infected population (asymptomatic + symptomatic).
        IFR: dict
            Rates of mortality by age group, type of hospitalization,
            and unit of rates, with type of hospitalization as
            primary key, measure unit as secondary key, and array of
            corresponding time series of rates as values.
            For example, mortality_rates['HICU']['per_capita'] is the time
            series of probability of death in ICU per capita among infected
            population (asymptomatic + symptomatic + hospitalized).
        """

        IHR = defaultdict(dict)
        IFR = defaultdict(dict)
        keys = ['HGen', 'HICU', 'HVent']
        for measure_unit in self.measure_units:
            IHR_by_unit = self._age_specific_IHR_by_hospitalization(measure_unit)
            for key in keys:
                IHR[key].update(IHR_by_unit[key])

        for measure_unit in self.measure_units:
            IFR_by_unit = self._age_specific_IFR_by_hospitalization(measure_unit, IHR)
            for key in keys:
                IFR[key].update(IFR_by_unit[key])

        return IHR, IFR

    def _calculate_age_specific_HR(self, measure, measure_unit):
        """
        Calculates age specific hospitalization rates given the specified
        covid measure and unit. The measure determines the types of
        hospitalizations to count and measure unit determines which
        hospitalization rates to use.

        Parameters
        ----------
        measure: CovidMeasure
            Specifies the measure to calculate.
        measure_unit: CovidMeasureUnit
            Specifies the measure's unit.

        Returns
        -------
          : np.array
            Array of time series of age-specific hospitalization rates of
            infected population or general population.
        """
        if measure is CovidMeasure.IHR:
            return (self.IHR['HGen'][measure_unit.value]
                  + self.IHR['HICU'][measure_unit.value]
                  + self.IHR['HVent'][measure_unit.value])

        elif measure is CovidMeasure.HR:
            # prevalence is used to count the probability that a person is
            # infected
            return (self.IHR['HGen'][measure_unit.value]
                  + self.IHR['HICU'][measure_unit.value]
                  + self.IHR['HVent'][measure_unit.value]) * self.prevalence

        elif measure is CovidMeasure.IHR_GENERAL:
            return self.IHR['HGen'][measure_unit.value]

        elif measure is CovidMeasure.HR_GENERAL:
            # prevalence is used to count the probability that a person is
            # infected
            return self.IHR['HGen'][measure_unit.value] * self.prevalence

        elif measure is CovidMeasure.IHR_ICU:
            return (self.IHR['HICU'][measure_unit.value]
                  + self.IHR['HVent'][measure_unit.value])

        elif measure is CovidMeasure.HR_ICU:
            # prevalence is used to count the probability that a person is
            # infected
            return (self.IHR['HICU'][measure_unit.value]
                  + self.IHR['HVent'][measure_unit.value]) * self.prevalence

        else:
            logging.warning(f'covid_measure {measure.value} is not relevant to hospitalization rate')
            return None

    def _calculate_age_specific_IFR(self, measure_unit):
        """
        Calculates age specific infection fatality rate (IFR).

        Parameters
        ----------
        measure_unit: CovidMeasureUnit
            Unit of IFR, determines how IFR is calculated.

        Returns
        -------
        IFR: np.array
            Array of time series of age-specific infection fatality rate.
        """

        IFR = 0
        for key in self.IFR:
            IFR += self.IFR[key][measure_unit.value]

        return IFR


    def generate_predictions(self):
        """
        Generates predictions of covid measures using the MLE model.

        Returns
        -------
        predictions: dict
            Contains:
            - compartments:
              - <compartment>: time series of population at a specific
                               infection states (susceptible, infected,
                               hospitalized, etc.) by demographic group
                               simulated by MLE model. Each time series is
                               recorded as pd.DataFrame, with dates of
                               prediction as index.
              name of compartments include: S - susceptible, E - exposed,
              A - asymptomatic, I - symptomatic, HGen - in non-ICU, HICU - in
              ICU, HVent - on ventilator, N - entire population.
            - <measure>:
              - <measure_unit>: time series of covid measures by demographic
                                group predicted using the MLE model.
                                The time series is recorded as pd.DataFrame,
                                with dates of prediction as index.
        """
        predictions = defaultdict(dict)
        t0_date = datetime.fromisoformat(self.fit_results['t0_date'])
        dates = [t0_date + timedelta(days=int(t)) for t in self.parameters['t_list']]
        age_groups = ['-'.join([str(int(tup[0])), str(int(tup[1]))]) for tup in
                      self.parameters['age_groups']]

        for c in self.predictions_by_age:
            predictions['compartments'][c] = pd.DataFrame(self.predictions_by_age[c].T,
                                                          columns=age_groups,
                                                          index=pd.DatetimeIndex(dates))
        # assuming a stable demographic distribution through time
        predictions['compartments']['N'] = \
            pd.DataFrame(np.tile(self.parameters['N'], (len(self.parameters['t_list']), 1)),
                         columns=age_groups,
                         index=pd.DatetimeIndex(dates))

        if self.measures is not None:
            for measure in self.measures:
                for measure_unit in self.measure_units:
                    if measure is CovidMeasure.IFR:
                        predictions[measure.value][measure_unit.value] = self._calculate_age_specific_IFR(measure_unit)
                    else:
                        predictions[measure.value][measure_unit.value] = self._calculate_age_specific_HR(measure,
                                                                                                         measure_unit)

                    predictions[measure.value][measure_unit.value] = \
                        pd.DataFrame(predictions[measure.value][measure_unit.value].T,
                                     columns=age_groups,
                                     index=pd.DatetimeIndex(dates))


        return predictions


    def map_to_target_population(self, predictions):
        """
        Maps the age-specific covid measures predicted by the MLE model to
        the target age distribution.

        Parameters
        ----------
        predictions: dict
            Contains:
            - compartments:
              - <compartment>: time series of population at a specific
                               infection state (susceptible, infected,
                               hospitalized, etc.) by demographic group
                               simulated by MLE model. Each time series is
                               recorded as pd.DataFrame, with dates of
                               prediction as index.
              name of compartment include: S - susceptible, E - exposed,
              A - asymptomatic, I - symptomatic, HGen - in non-ICU, HICU - in
              ICU, HVent - on ventilator, N - entire population.
            - <measure>:
              - <measure_unit>: time series of covid measures by demographic
                                group predicted using the MLE model.
                                The time series is recorded as pd.DataFrame,
                                with dates of prediction as index.

        Returns
        -------
          : dict
            Contains:
            - compartments:
              - <compartment>: time series of population at a specific
                               infection state (susceptible, infected,
                               hospitalized, etc.) simulated by MLE model
                               assuming the population has the demographic
                               distribution of the target population. Each time
                               series is recorded as pd.DataFrame, with dates of
                               prediction as index.
              name of compartment include: S - susceptible, E - exposed,
              A - asymptomatic, I - symptomatic, HGen - in non-ICU, HICU - in ICU,
              HVent - on ventilator, N - entire population.
            - <measure>:
              - <measure_unit>: time series of covid measures predicted using
                                the MLE model and averaged over target
                                demographic distribution (adjusted by risk
                                modification if relative risk is specified).
                                The time series is recorded as pd.DataFrame,
                                with dates of prediction as index.
        """
        # calculate weights
        age_bin_centers = [np.mean(tup) for tup in self.parameters['age_groups']]
        weights = self.target_age_distribution_pdf(age_bin_centers)
        if (weights != 1).sum() == 0:
            logging.warning('no target age distribution is given, measure is aggregated assuming age '
                            'distrubtion at given FIPS code')

        weights /= weights.sum()
        demographic_group_size_ratio = weights / (self.parameters['N'] / self.parameters['N'].sum())

        mapped_predictions = defaultdict(dict)

        for c in predictions['compartments']:
            mapped_predictions['compartments'][c] = predictions['compartments'][c].dot(demographic_group_size_ratio)

        measure_names = [k for k in predictions.keys() if k != 'compartments']
        if len(measure_names) > 0:
            for measure_name in measure_names:
                for measure_unit_name in predictions[measure_name]:
                    modified_weights = weights
                    if self.risk_modifier_by_age is not None:
                        if measure_name in self.risk_modifier_by_age:
                            modified_weights = weights * self.risk_modifier_by_age[measure_name](age_bin_centers)
                            modified_weights /= modified_weights.sum()
                    mapped_predictions[measure_name][measure_unit_name] = \
                        predictions[measure_name][measure_unit_name].dot(modified_weights)

        self.results = mapped_predictions

        return self.results

    def run(self):
        """
        Makes predictions of age-specific population size at each state of
        infection and covid measures using the MLE model and maps them to
        the target age distribution.

        Returns
        -------
          : dict
            Contains:
            - compartments:
              - <compartment>: time series of population at a specific
                               infection state (susceptible, infected,
                               hospitalized, etc.) simulated by MLE model
                               assuming the population has the demographic
                               distribution of the target population. Each time
                               series is recorded as pd.DataFrame, with dates of
                               prediction as index.
              name of compartment include: S - susceptible, E - exposed,
              A - asymptomatic, I - symptomatic, HGen - in non-ICU, HICU - in
              ICU, HVent - on ventilator, N - entire population.
            - <measure>:
              - <measure_unit>: time series of covid measures predicted using
                                the MLE model and averaged over target
                                demographic distribution (adjusted by risk
                                modification if relative risk is specified).
                                The time series is recorded as pd.DataFrame,
                                with dates of prediction as index.
        """
        predictions = self.generate_predictions()
        self.results = self.map_to_target_population(predictions)
        return self.results
