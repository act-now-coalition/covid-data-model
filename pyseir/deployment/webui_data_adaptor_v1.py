import os
from pyseir import OUTPUT_DIR
import numpy as np
import math
import pandas as pd
from datetime import timedelta, datetime, date
from pyseir import load_data
import simplejson as json
import logging
import us
from multiprocessing import Pool
from libs.datasets import FIPSPopulation, JHUDataset, CDSDataset
from libs.datasets.dataset_utils import build_aggregate_county_data_frame
from libs.datasets.dataset_utils import AggregationLevel
from pyseir.ensembles.ensemble_runner import EnsembleRunner


class WebUIDataAdaptorV1:
    """
    Map pyseir output data model to the format required by CovidActNow's webUI.

    Parameters
    ----------
    state: str
        State to map outputs for.
    """
    def __init__(self, state, output_interval_days=4, run_mode='can-before',
                 output_dir=None, jhu_dataset=None, cds_dataset=None):

        self.output_interval_days = output_interval_days
        self.state = state
        self.run_mode = run_mode

        output_dir = output_dir or OUTPUT_DIR
        self.county_output_dir = os.path.join(output_dir, 'web_ui', 'county')
        self.state_output_dir = os.path.join(output_dir, 'web_ui', 'state')
        self.state_abbreviation = us.states.lookup(state).abbr

        os.makedirs(self.county_output_dir, exist_ok=True)
        os.makedirs(self.state_output_dir, exist_ok=True)
        self.population_data = FIPSPopulation.local().population()

        self.jhu_local = jhu_dataset or JHUDataset.local()
        self.cds_dataset = cds_dataset or CDSDataset.local()

        self.county_timeseries = build_aggregate_county_data_frame(self.jhu_local, self.cds_dataset)
        self.county_timeseries['date'] = self.county_timeseries['date'].dt.normalize()

        self.state_timeseries = self.jhu_local.timeseries().state_data
        self.state_timeseries['date'] = self.state_timeseries['date'].dt.normalize()

    def backfill_output_model_fips(self, fips, t0, final_beds, output_model):
        """
        Add backfilled hospitalization, case, amd deaths data.

        Parameters
        ----------
        fips: str
            State or county fips code.
        t0: datetime
            Start time for the simulation.
        final_beds: total number of beds.
            Number of beds after scaling.
        output_model: dict
            Output model to impute.

        Returns
        -------
        backfill: str
            Backfill dataframe.
        """
        backfill_to_date = date(2020, 3, 3)   # @TODO: Parameterize
        hospitalization_rate = 0.04          # @TODO: Parameterize
        intervals_to_backfill = math.ceil((t0.date() - backfill_to_date).days / self.output_interval_days)
        backfill_offsets = range(-intervals_to_backfill * self.output_interval_days, 0, self.output_interval_days)

        backfill = pd.DataFrame()
        backfill['days'] = backfill_offsets
        backfill['date'] = [(t0 + timedelta(days=t)) for t in backfill_offsets]
        backfill['date'] = backfill['date'].dt.normalize()
        backfill['beds'] = final_beds

        if len(fips) == 5:
            actual_timeseries = self.county_timeseries[(self.county_timeseries['fips'] == fips)]
        else:
            actual_timeseries = self.state_timeseries[(self.state_timeseries['state'] == self.state_abbreviation)]

        backfill = pd.merge(backfill, actual_timeseries[['date', 'cases', 'deaths']], on='date', how='left')

        # TODO this is fragile because of the backfill hospitalization date
        #      alignment and/or 4 day discontinuity. Luckily it is also invisible on non-log-scales.
        # Account for two cases
        #   (i) hospital admissions available.
        #   (ii) Not available, so cases are imputed...
        # We can just read the initial conditions infected and hospitalized to rescale the case data to match.
        backfill['all_infected'] =( output_model['all_infected'][0] * backfill['cases'] / backfill['cases'].max()).fillna(0)
        backfill['all_hospitalized'] = np.multiply(backfill['all_infected'], hospitalization_rate).fillna(0)

        backfill['dead'] = backfill['deaths'].fillna(0)
        backfill['date'] = backfill['date'].dt.strftime('%m/%d/%y')

        return backfill

    def map_fips(self, fips):
        """
        For a given county fips code, generate the CAN UI output format.

        Parameters
        ----------
        fips: str
            County FIPS code to map.
        """
        if len(fips) == 5:
            population = self.population_data.get_county_level('USA', state=self.state_abbreviation, fips=fips)
            output_dir = self.county_output_dir
        else:
            population = self.population_data.get_state_level('USA', state=self.state_abbreviation)
            output_dir = self.state_output_dir

        logging.info(f'Mapping output to WebUI for {self.state}, {fips}')
        pyseir_outputs = load_data.load_ensemble_results(fips, run_mode=self.run_mode)

        t0 = datetime.today()

        policies = [key for key in pyseir_outputs.keys() if key.startswith('suppression_policy')]
        for i_policy, suppression_policy in enumerate(policies):
            output_for_policy = pyseir_outputs[suppression_policy]
            output_model = pd.DataFrame()

            t_list = output_for_policy['t_list']
            t_list_downsampled = range(0, int(max(t_list)), self.output_interval_days)
            # Col 0
            output_model['days'] = t_list_downsampled
            # Col 1
            output_model['date'] = [(t0 + timedelta(days=t)).date().strftime('%m/%d/%y') for t in t_list_downsampled]
            # Col 2
            output_model['t'] = population
            # Col 3
            output_model['b'] = np.interp(t_list_downsampled, t_list, output_for_policy['S']['ci_50'])
            # Col 4
            output_model['c'] = np.interp(t_list_downsampled, t_list, output_for_policy['E']['ci_50'])
            # Col 5
            output_model['d'] = np.interp(t_list_downsampled, t_list, np.add(output_for_policy['I']['ci_50'], output_for_policy['A']['ci_50'])) # Infected + Asympt.
            # Col 6
            output_model['e'] = output_model['d']
            # Col 7
            output_model['f'] = np.interp(t_list_downsampled, t_list, output_for_policy['HGen']['ci_50']) # Hosp General
            # Col 8
            output_model['g'] = np.interp(t_list_downsampled, t_list, output_for_policy['HICU']['ci_50']) # Hosp ICU
            # Col 9
            output_model['all_hospitalized'] = np.add(output_model['f'], output_model['g'])
            # Col 10
            output_model['all_infected'] = output_model['d']
            # Col 11
            output_model['dead'] = np.interp(t_list_downsampled, t_list, output_for_policy['total_deaths']['ci_50'])
            # Col 12
            final_beds = np.mean(output_for_policy['HGen']['capacity']) + np.mean(output_for_policy['HICU']['capacity'])
            output_model['beds'] = final_beds

            backfill = self.backfill_output_model_fips(fips, t0, final_beds, output_model)
            output_model = pd.concat([backfill, output_model])[output_model.columns].reset_index(drop=True)

            # Truncate date range of output.
            output_dates = pd.to_datetime(output_model['date'])
            output_model = output_model[ (output_dates > datetime(month=3, day=3, year=2020))
                                        & (output_dates < datetime.today() + timedelta(days=90))]
            output_model = output_model.fillna(0)

            for col in ['i', 'j', 'k', 'l']:
                output_model[col] = 0
            output_model['population'] = population
            for col in ['m', 'n']:
                output_model[col] = 0

            # Truncate floats and cast as strings to match data model.
            int_columns = [col for col in output_model.columns if col not in ('date')]
            output_model[int_columns] = output_model[int_columns].fillna(0).astype(int).astype(str)

            # Convert the records format to just list(list(values))
            output_model = [[val for val in timestep.values()] for timestep in output_model.to_dict(orient='records')]

            if len(fips) == 5:
                output_path = os.path.join(output_dir, f'{self.state_abbreviation}.{fips}.{i_policy}.json')
            else:
                output_path = os.path.join(output_dir, f'{self.state_abbreviation}.{i_policy}.json')

            with open(output_path, 'w') as f:
                json.dump(output_model, f)

    def generate_state(self, states_only=False, include_imputed=False):
        """
        Generate for each county in a state, the output for the webUI.

        Parameters
        ----------
        states_only: bool
            If True only run the state level.
        include_imputed:
            If True, map the outputs for imputed counties as well as those with
            no data.
        """
        state_fips = us.states.lookup(self.state).fips
        self.map_fips(state_fips)

        if not states_only:
            df = load_data.load_county_metadata()
            all_fips = df[df['state'].str.lower() == self.state.lower()].fips

            if not include_imputed:
                # Filter...
                fips_with_data = self.jhu_local.timeseries() \
                    .get_subset(AggregationLevel.COUNTY, country='USA') \
                    .get_data(country='USA', state=self.state_abbreviation).fips.unique().tolist()

                all_fips = [fips for fips in all_fips if fips in fips_with_data]
            p = Pool()
            p.map(self.map_fips, all_fips)
            p.close()


if __name__ == '__main__':
    mapper = WebUIDataAdaptorV1('California', output_interval_days=4)
    mapper.generate_state()
