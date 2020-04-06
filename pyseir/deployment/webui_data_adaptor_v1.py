import os
from pyseir import OUTPUT_DIR
import numpy as np
import pandas as pd
from datetime import timedelta
from pyseir import load_data
import json
import logging
import us
from multiprocessing import Pool


class WebUIDataAdaptorV1:
    """
    Map pyseir output data model to the format required by CovidActNow's webUI.

    Parameters
    ----------
    state: str
        State to map outputs for.
    """

    def __init__(self, state, output_interval_days=4, run_mode='can-before'):

        self.output_interval_days = output_interval_days
        self.state = state
        self.run_mode = run_mode
        self.county_output_dir = os.path.join(OUTPUT_DIR, 'web_ui', 'county')
        self.state_output_dir = os.path.join(OUTPUT_DIR, 'web_ui', 'state')
        self.state_abbreviation = us.states.lookup(state).abbr

        os.makedirs(self.county_output_dir, exist_ok=True)
        os.makedirs(self.state_output_dir, exist_ok=True)

    def map_fips(self, fips):
        """
        For a given county fips code, generate the CAN UI output format.

        Parameters
        ----------
        fips: str
            County FIPS code to map.
        """
        county_metadata = load_data.load_county_metadata_by_fips(fips)
        logging.info(f'Mapping output to WebUI for {county_metadata["county"]}, {county_metadata["state"]}')
        pyseir_outputs = load_data.load_ensemble_results(fips, run_mode=self.run_mode)
        import datetime
        t0 = datetime.datetime.today()  #fit_results.load_t0(fips)

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
            output_model['t'] = county_metadata['total_population']
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
            output_model['beds'] = np.mean(output_for_policy['HGen']['capacity']) + np.mean(output_for_policy['HICU']['capacity'])

            for col in ['i', 'j', 'k', 'l']:
                output_model[col] = 0
            output_model['population'] = county_metadata['total_population']
            for col in ['m', 'n']:
                output_model[col] = 0

            # Truncate floats and cast as strings to match data model.
            int_columns = [col for col in output_model.columns if col not in ('date')]
            output_model[int_columns] = output_model[int_columns].fillna(0).astype(int).astype(str)

            # Convert the records format to just list(list(values))
            output_model = [[val for val in timestep.values()] for timestep in output_model.to_dict(orient='records')]

            with open(os.path.join(self.county_output_dir, f'{self.state_abbreviation}.{fips}.{i_policy}.json'), 'w') as f:
                json.dump(output_model, f)

    def generate_state(self):
        """
        Generate for each county in a state, the output for the webUI.
        """
        df = load_data.load_county_metadata()
        all_fips = df[df['state'].str.lower() == self.state.lower()].fips
        p = Pool()
        p.map(self.map_fips, all_fips)
        p.close()


if __name__ == '__main__':
    mapper = WebUIDataAdaptorV1('California', output_interval_days=4)
    mapper.generate_state()
