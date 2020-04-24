import us
import logging
import simplejson as json
from multiprocessing import Pool
from pyseir.inference import fit_results
from pyseir import load_data
from pyseir.utils import RunArtifact, get_run_artifact_path

json.encoder.FLOAT_REPR = lambda o: format(o, '.4f')


class OutputIntegrator:

    def __init__(self, state, output_dir):

        self.state = state
        self.state_abbreviation = us.states.lookup(state).abbr
        self.state_fips = us.states.lookup(state).fips
        self.output_dir = output_dir

    def map_fips(self, fips):
        """
        Merge all model data into a fused output model.

        Parameters
        ----------
        fips: str
            State or county FIPS code.
        """
        logging.info(f'Merging outputs for {self.state} ({fips})')
        output = dict()

        output['models'] = load_data.load_ensemble_results(fips)
        output['indicators'] = dict()

        try:
            output['inference_fit'] = fit_results.load_inference_result(fips)

            for k, v in output['inference_fit'].items():
                if type(v).__module__ == 'numpy':
                    output['inference_fit'][k] = v.item()
        except (KeyError, ValueError):
            logging.warning(f'No inference result found for {fips}. Skipping...')
            output['inference_fit'] = dict()

        try:
            df = fit_results.load_Rt_result(fips)
            df['date'] = [d.date().isoformat() for d in df.index.to_pydatetime()]
            output['indicators']['Rt'] = df.to_dict(orient='list')
        except ValueError:
            logging.warning(f'No Rt result found for {fips}. Skipping...')
            output['indicators']['Rt'] = dict()

        if len(fips) == 5:
            output['county_metadata'] = load_data.load_county_metadata_by_fips(fips)

        output_path = get_run_artifact_path(
            fips=fips,
            artifact=RunArtifact.MERGED_OUTPUT,
            output_dir=self.output_dir)

        with open(output_path, 'w') as f:
            json.dump(output, f)

    def generate_state(self, states_only=False):
        """
        Generate for each county in a state, the output for the webUI.

        Parameters
        ----------
        states_only: bool
            If True only run the state level.
        """
        self.map_fips(self.state_fips)

        if not states_only:
            df = load_data.load_county_metadata()
            all_fips = df[df['state'].str.lower() == self.state.lower()].fips

        p = Pool()
        p.map(self.map_fips, all_fips)
        p.close()
