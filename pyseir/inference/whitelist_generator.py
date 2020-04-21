import pandas as pd
import numpy as np
from pyseir import load_data
from datetime import datetime
from pyseir.utils import get_run_artifact_path, RunArtifact


class WhitelistGenerator:
    """
    This class applies filters to inference results to determine a set of
    Counties to whitelist in the API.

    Parameters
    ----------
    total_cases: int
        Min number of total cases to whitelist.
    total_deaths: int
        Min number of total cases to whitelist.
    nonzero_case_datapoints: int
        Nonzero case datapoints.
    nonzero_death_datapoints: int
        Minimum number of nonzero death datapoints in the time series to allow
        display.
    """
    def __init__(
            self,
            total_cases=100,
            total_deaths=10,
            nonzero_case_datapoints=20,
            nonzero_death_datapoints=2):
        self.county_metadata = load_data.load_county_metadata()
        self.df_whitelist = None

        self.total_cases = total_cases
        self.total_deaths = total_deaths
        self.nonzero_case_datapoints = nonzero_case_datapoints
        self.nonzero_death_datapoints = nonzero_death_datapoints

    def generate_whitelist(self):
        """
        Generate a county whitelist based on the cuts above.

        Returns
        -------
        df: whitelist
        """
        whitelist_generator_inputs = []
        for fips in self.county_metadata.fips:
            times, observed_new_cases, observed_new_deaths = load_data.load_new_case_data_by_fips(
                fips, t0=datetime(day=1, month=1, year=2020))

            metadata = self.county_metadata[self.county_metadata.fips == fips].iloc[0].to_dict()

            record = dict(
                fips=fips,
                state=metadata['state'],
                county=metadata['county'],
                total_cases=observed_new_cases.sum(),
                total_deaths=observed_new_deaths.sum(),
                nonzero_case_datapoints=np.sum(observed_new_cases > 0),
                nonzero_death_datapoints=np.sum(observed_new_deaths > 0)
            )
            whitelist_generator_inputs.append(record)

        df_candidates = pd.DataFrame(whitelist_generator_inputs)

        df_whitelist = df_candidates[['fips', 'state', 'county']]

        df_whitelist['inference_ok'] = (
                  (df_candidates.nonzero_case_datapoints >= self.nonzero_case_datapoints)
                & (df_candidates.nonzero_death_datapoints >= self.nonzero_death_datapoints)
                & (df_candidates.total_cases >= self.total_cases)
                & (df_candidates.total_deaths >= self.total_deaths)
        )

        # Dummy fips since not used here...
        output_path = get_run_artifact_path(fips='06', artifact=RunArtifact.WHITELIST_RESULT)
        df_whitelist.to_json(output_path)

        return df_whitelist
