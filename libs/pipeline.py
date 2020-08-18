"""
Code that is used to help move information around in the pipeline, starting with `Region` which
represents a geographical area (state, county, metro area, etc).
"""


import json
import os
from typing import Optional

import pandas as pd
from pydantic import BaseModel

import pyseir
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import FIPSPopulation
from libs.datasets.combined_datasets import load_us_latest_dataset, _log
from pyseir.rt.utils import NEW_ORLEANS_FIPS
from pyseir.utils import get_run_artifact_path, RunArtifact


class Region(BaseModel):
    """Identifies and provides access to data about a geographical area."""

    # The FIPS identifier for the region, either 2 digits for a state or 5 digits for a county.
    fips: str

    def __repr__(self) -> str:
        return f"Region(fips={self.fips})"

    def get_us_latest(self):
        """Gets latest values for a given state or county fips code."""
        us_latest = load_us_latest_dataset()
        return us_latest.get_record_for_fips(self.fips)

    def load_inference_result(self):
        """
        Load fit results by state or county fips code.

        Returns
        -------
        : dict
            Dictionary of fit result information.
        """
        output_file = get_run_artifact_path(self.fips, RunArtifact.MLE_FIT_RESULT)
        df = pd.read_json(output_file, dtype={"fips": "str"})
        if len(self.fips) == 2:
            return df.iloc[0].to_dict()
        else:
            return df.set_index("fips").loc[self.fips].to_dict()

    def get_population(self) -> int:
        """Gets the population for this region."""
        population_data = FIPSPopulation.local().population()
        return population_data.get_record_for_fips(self.fips)[CommonFields.POPULATION]

    def load_ensemble_results(self) -> Optional[dict]:
        """Retrieves ensemble results for this region."""
        output_filename = pyseir.utils.get_run_artifact_path(
            self.fips, pyseir.utils.RunArtifact.ENSEMBLE_RESULT
        )
        if not os.path.exists(output_filename):
            return None

        with open(output_filename) as f:
            return json.load(f)

    def load_Rt_result(self) -> Optional[pd.DataFrame]:
        """Loads the Rt inference result.

        Returns
        -------
        results: pd.DataFrame
            DataFrame containing the R_t inferences.
        """
        if self.fips in NEW_ORLEANS_FIPS:
            _log.info("Applying New Orleans Patch")
            return pyseir.rt.patches.patch_aggregate_rt_results(NEW_ORLEANS_FIPS)

        path = pyseir.utils.get_run_artifact_path(
            self.fips, pyseir.utils.RunArtifact.RT_INFERENCE_RESULT
        )
        if not os.path.exists(path):
            return None
        return pd.read_json(path)

    @staticmethod
    def from_fips(fips: str):
        return Region(fips=fips)
