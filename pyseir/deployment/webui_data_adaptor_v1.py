import numpy as np
import math
import pandas as pd
import simplejson as json
import logging
import us
from datetime import timedelta, datetime, date
from multiprocessing import Pool
from pyseir import load_data
from pyseir.inference.fit_results import load_inference_result, load_Rt_result
from pyseir.utils import get_run_artifact_path, RunArtifact, RunMode
from libs.enums import Intervention
from libs.datasets import FIPSPopulation, JHUDataset, CDSDataset
from libs.datasets.dataset_utils import build_aggregate_county_data_frame
from libs.datasets.dataset_utils import AggregationLevel
import libs.datasets.can_model_output_schema as schema


class WebUIDataAdaptorV1:
    """
    Map pyseir output data model to the format required by CovidActNow's webUI.

    Parameters
    ----------
    state: str
        State to map outputs for.
    include_imputed:
        If True, map the outputs for imputed counties as well as those with
        no data.
    """
    def __init__(self, state, output_interval_days=4, run_mode='can-before',
                 output_dir=None, jhu_dataset=None, cds_dataset=None, include_imputed=False):

        self.output_interval_days = output_interval_days
        self.state = state
        self.run_mode = RunMode(run_mode)
        self.include_imputed = include_imputed
        self.state_abbreviation = us.states.lookup(state).abbr
        self.population_data = FIPSPopulation.local().population()
        self.output_dir = output_dir

        self.jhu_local = jhu_dataset or JHUDataset.local()
        self.cds_dataset = cds_dataset or CDSDataset.local()

        self.county_timeseries = build_aggregate_county_data_frame(self.jhu_local, self.cds_dataset)
        self.county_timeseries['date'] = self.county_timeseries['date'].dt.normalize()

        self.state_timeseries = self.jhu_local.timeseries().state_data
        self.state_timeseries['date'] = self.state_timeseries['date'].dt.normalize()
        self.df_whitelist = load_data.load_whitelist()
        self.df_whitelist = self.df_whitelist[self.df_whitelist['inference_ok'] == True]

    def map_fips(self, fips):
        """
        For a given county fips code, generate the CAN UI output format.

        Parameters
        ----------
        fips: str
            County FIPS code to map.
        """
        logging.info(f'Mapping output to WebUI for {self.state}, {fips}')
        pyseir_outputs = load_data.load_ensemble_results(fips)

        # Fit results not always available if the fit failed, or there are no
        # inference results. These do always exist for states however.
        try:
            fit_results = load_inference_result(fips)
            t0_simulation = datetime.fromisoformat(fit_results['t0_date'])
        except (KeyError, ValueError):
            fit_results = None
            state_fit_results = load_inference_result(fips[:2])
            t0_simulation = datetime.fromisoformat(state_fit_results['t0_date'])
            logging.error(f'Fit result not found for {fips}. Skipping...}')

        hosp_times, current_hosp, _ = load_data.load_hospitalization_data_by_state(
            state=self.state_abbreviation,
            t0=t0_simulation,
            convert_cumulative_to_current=True)
        t_latest_hosp_data, current_hosp = hosp_times[-1], current_hosp[-1]

        if len(fips) == 5:
            population = self.population_data.get_county_level('USA', state=self.state_abbreviation, fips=fips)
            state_population = self.population_data.get_state_level('USA', state=self.state_abbreviation)
            # Rescale hosps based on the population ratio... Could swap this to infection ratio later?
            current_hosp *= population / state_population
        else:
            population = self.population_data.get_state_level('USA', state=self.state_abbreviation)

        policies = [key for key in pyseir_outputs.keys() if key.startswith('suppression_policy')]

        # Don't ship counties that are not whitelisted.
        for i_policy, suppression_policy in enumerate(policies):
            if (len(fips) == 5 and fips not in self.df_whitelist.fips.values) or fit_results is None:
                continue

            output_for_policy = pyseir_outputs[suppression_policy]
            output_model = pd.DataFrame()

            total_hosps = output_for_policy['HGen']['ci_50'][t_latest_hosp_data] + output_for_policy['HICU']['ci_50'][t_latest_hosp_data]
            hosp_fraction = current_hosp / total_hosps

            t_list = output_for_policy['t_list']
            t_list_downsampled = range(0, int(max(t_list)), self.output_interval_days)

            output_model[schema.DAY_NUM] = t_list_downsampled
            output_model[schema.DATE] = [(t0_simulation + timedelta(days=t)).date().strftime('%m/%d/%y') for t in t_list_downsampled]
            output_model[schema.TOTAL] = population
            output_model[schema.TOTAL_SUSCEPTIBLE] = np.interp(t_list_downsampled, t_list, output_for_policy['S']['ci_50'])
            output_model[schema.EXPOSED] = np.interp(t_list_downsampled, t_list, output_for_policy['E']['ci_50'])
            output_model[schema.INFECTED] = np.interp(t_list_downsampled, t_list, np.add(output_for_policy['I']['ci_50'], output_for_policy['A']['ci_50'])) # Infected + Asympt.
            output_model[schema.INFECTED_A] = output_model[schema.INFECTED]
            output_model[schema.INFECTED_B] = hosp_fraction * np.interp(t_list_downsampled, t_list, output_for_policy['HGen']['ci_50']) # Hosp General
            output_model[schema.INFECTED_C] = hosp_fraction * np.interp(t_list_downsampled, t_list, output_for_policy['HICU']['ci_50']) # Hosp ICU
            # General + ICU beds. don't include vent here because they are also counted in ICU
            output_model[schema.ALL_HOSPITALIZED] = np.add(output_model[schema.INFECTED_B], output_model[schema.INFECTED_C])
            output_model[schema.ALL_INFECTED] = output_model[schema.INFECTED]
            output_model[schema.DEAD] = np.interp(t_list_downsampled, t_list, output_for_policy['total_deaths']['ci_50'])
            final_beds = np.mean(output_for_policy['HGen']['capacity'])
            output_model[schema.BEDS] = final_beds
            output_model[schema.CUMULATIVE_INFECTED] = np.interp(t_list_downsampled, t_list, np.cumsum(output_for_policy['total_new_infections']['ci_50']))

            if fit_results:
                output_model[schema.Rt] = np.interp(t_list_downsampled, t_list, fit_results['eps'] * fit_results['R0'] * np.ones(len(t_list)))
                output_model[schema.Rt_ci90] = np.interp(t_list_downsampled, t_list, 2 * fit_results['eps_error'] * fit_results['R0'] * np.ones(len(t_list)))
            else:
                output_model[schema.Rt] = 0
                output_model[schema.Rt_ci90] = 0

            output_model[schema.CURRENT_VENTILATED] = hosp_fraction * np.interp(t_list_downsampled, t_list, output_for_policy['HVent']['ci_50'])
            output_model[schema.POPULATION] = population
            output_model[schema.ICU_BED_CAPACITY] = np.mean(output_for_policy['HICU']['capacity'])
            output_model[schema.VENTILATOR_CAPACITY] = np.mean(output_for_policy['HVent']['capacity'])

            # Truncate date range of output.
            output_dates = pd.to_datetime(output_model['date'])
            output_model = output_model[ (output_dates >= datetime(month=3, day=3, year=2020))
                                        & (output_dates < datetime.today() + timedelta(days=90))]
            output_model = output_model.fillna(0)

            try:
                rt_results = load_Rt_result(fips)
                rt_results.index = rt_results['Rt_MAP_composite'].index.strftime('%m/%d/%y')
                merged = output_model.merge(rt_results[['Rt_MAP_composite', 'Rt_ci95_composite']],
                    right_index=True, left_on='date', how='left')

                output_model[schema.RT_INDICATOR] = merged['Rt_MAP_composite']
                # With 90% probability the value is between rt_indicator - ci90 to rt_indicator + ci90
                output_model[schema.RT_INDICATOR_CI90] = merged['Rt_ci95_composite'] - merged['Rt_MAP_composite']
            except (ValueError, KeyError) as e:
                output_model[schema.RT_INDICATOR] = "NaN"
                output_model[schema.RT_INDICATOR_CI90] = "NaN"

            output_model[[schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]] = \
                output_model[[schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]].fillna("NaN")

            # Truncate floats and cast as strings to match data model.
            int_columns = [col for col in output_model.columns if col not in
                           (schema.DATE, schema.Rt, schema.Rt_ci90, schema.RT_INDICATOR, schema.RT_INDICATOR_CI90)]
            output_model.loc[:, int_columns] = output_model[int_columns].fillna(0).astype(int).astype(str)
            output_model.loc[:, [schema.Rt, schema.Rt_ci90, schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]] = \
                output_model[[schema.Rt, schema.Rt_ci90, schema.RT_INDICATOR, schema.RT_INDICATOR_CI90]]\
                    .fillna(0).round(decimals=4).astype(str)

            # Convert the records format to just list(list(values))
            output_model = [[val for val in timestep.values()] for timestep in output_model.to_dict(orient='records')]

            output_path = get_run_artifact_path(fips, RunArtifact.WEB_UI_RESULT, output_dir=self.output_dir)
            policy_enum = Intervention.from_webui_data_adaptor(suppression_policy)
            output_path = output_path.replace('__INTERVENTION_IDX__', str(policy_enum.value))
            with open(output_path, 'w') as f:
                json.dump(output_model, f)

    def generate_state(self, states_only=False):
        """
        Generate for each county in a state, the output for the webUI.

        Parameters
        ----------
        states_only: bool
            If True only run the state level.
        """
        state_fips = us.states.lookup(self.state).fips
        self.map_fips(state_fips)

        if not states_only:
            df = load_data.load_county_metadata()
            all_fips = df[df['state'].str.lower() == self.state.lower()].fips

            if not self.include_imputed:
                # Filter...
                fips_with_cases = self.jhu_local.timeseries() \
                    .get_subset(AggregationLevel.COUNTY, country='USA') \
                    .get_data(country='USA', state=self.state_abbreviation)
                fips_with_cases = fips_with_cases[fips_with_cases.cases > 0].fips.unique().tolist()
                all_fips = [fips for fips in all_fips if fips in fips_with_cases]

            p = Pool()
            p.map(self.map_fips, all_fips)
            p.close()


if __name__ == '__main__':
    mapper = WebUIDataAdaptorV1('California', output_interval_days=4)
    mapper.generate_state()
