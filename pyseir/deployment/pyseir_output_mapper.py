import os
from pyseir import OUTPUT_DIR
import numpy as np
import pandas as pd
from datetime import timedelta
from pyseir import load_data
from pyseir.inference import fit_results
import json
import us
from multiprocessing import Pool


class PyseirOutputToWebUI:
    """
    Map pyseir output data model to the format required by CovidActNow's webUI.

    Parameters
    ----------
    state: str
        State to map outputs for.
    """

    def __init__(self, state, output_interval_days=4):

        self.output_interval_days = output_interval_days
        self.state = state
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
        pyseir_outputs = load_data.load_ensemble_results(fips)
        t0 = fit_results.load_t0(fips)

        for i_policy, (suppression_policy, output_for_policy) in enumerate(pyseir_outputs.items()):
            output_model = pd.DataFrame()

            t_list = output_for_policy['t_list']
            t_list_downsampled = range(0, int(max(t_list)), self.output_interval_days)

            output_model['days'] = t_list_downsampled
            output_model['date'] = [(t0 + timedelta(days=t)).date().strftime('%m/%d/%y') for t in t_list_downsampled]
            output_model['t'] = county_metadata['total_population']
            output_model['b'] = np.interp(t_list_downsampled, t_list, output_for_policy['S']['ci_50'])
            output_model['c'] = np.interp(t_list_downsampled, t_list, output_for_policy['E']['ci_50'])
            output_model['d'] = np.interp(t_list_downsampled, t_list, np.add(output_for_policy['I']['ci_50'], output_for_policy['A']['ci_50'])) # Infected + Asympt.
            output_model['e'] = output_model['d']
            output_model['f'] = np.interp(t_list_downsampled, t_list, output_for_policy['HGen']['ci_50']) # Hosp General
            output_model['g'] = np.interp(t_list_downsampled, t_list, output_for_policy['HICU']['ci_50']) # Hosp General
            output_model['all_hospitalized'] = np.add(output_model['f'], output_model['g'])
            output_model['all_infected'] = np.add(output_model['f'], output_model['g'])
            output_model['dead'] = np.interp(t_list_downsampled, t_list, output_for_policy['total_deaths']['ci_50'])
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
    mapper = PyseirOutputToWebUI('California', output_interval_days=4)
    mapper.generate_state()
